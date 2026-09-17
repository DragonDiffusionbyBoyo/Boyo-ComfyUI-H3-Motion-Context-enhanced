"""Pin previous-clip motion at the head of an H3 clip.

Wire it between a stock H3 conditioning node and the sampler:

    MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo (or the t2v path)
        -> H3 Motion Context
        -> guider / sampler

Two axes to test, both cheap.

encode_mode
  frames  one VAE call per frame, each pinned as its own cond block. The
          model sees N snapshots at N instants.
  video   one VAE call for the whole run. The H3 video VAE has latent_dim
          3, so it reads the batch axis as time and compresses the run
          into fewer latent steps (5 pixel frames -> 2 steps, 22 -> 7).
          Each step becomes one cond block, so the motion between frames
          lives inside the latent instead of being implied across separate
          stills. Far fewer rows and one VAE load.

anchor_mode
  head    pinned frames occupy indices 0..N-1 of the delivered timeline.
          They come back in the output, so trim that many frames off the
          front before concatenating.
  before  pinned frames sit at negative indices, ending at -1, so
          delivered frame 0 continues from them and nothing is wasted.
          Their time coordinates land below text_len, which is the range
          the text rows occupy. Whether that collision matters is exactly
          what this mode is asking.
"""

import gc
import logging
import os
import re
import shutil
import subprocess
import tempfile

import comfy.utils
import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI always ships safetensors; belt and braces
    _st_load = _st_save = None

from .layout_contract import ensure as _ensure_layout_contract
from .layout_contract import is_checked as _layout_checked

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context")


def _ensure_layout_ok():
    """Prove ComfyUI still places anchors the way this pack needs, once.

    This used to install two runtime patches. ComfyUI 0.33 does natively
    what they existed to do, so the node now builds plain keyframe dicts and
    only has to check that the arithmetic behind them still holds. See
    layout_contract.py for what is checked and why it is not free.

    Run on first use rather than at import, same as the patches were: the
    pack sitting in custom_nodes should change nothing at all until you
    actually chain a clip. The cost is that a failure shows up on the
    first render instead of in the startup log.
    """
    _ensure_layout_contract("pinning a clip")


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# Run lengths the video VAE's downscale formula max(1, (n - 5) // 17 * 5 + 2)
# actually distinguishes. Anything between two grid points encodes to the same
# number of latent steps as the lower one, but the steps then cover the FIRST
# `covered` frames of the input rather than the last: encoding 10 frames yields
# the same 2 steps as encoding 5, representing frames [-10..-6] of the source
# clip instead of [-5..-1]. The pinned run would end five frames early and the
# delivered clip would continue from the wrong instant. So off-grid requests
# are snapped DOWN before slicing, keeping content and coverage in agreement.
# The grid is 17m+5 and continues upward; the node only offers up to 56,
# but the snap-down logic knows the higher points so an out-of-range
# request lands on the nearest real one instead of being clamped to 39.
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)

# Settings that used to be widgets. Each had exactly one right answer, so
# offering the wrong one was noise. The losing branches are still in the
# code below: change a constant here to reproduce the failure they cause.
#
#   ENCODE_MODE   "video" encodes the pinned run in one VAE call, so the
#                 motion lives inside the latent. "frames" encodes each
#                 frame as its own still, costs twice the rows and left a
#                 visible seam in testing.
#   ANCHOR_MODE   "head" pins the run at the start of the clip, where the
#                 Trim node removes it. "before" places it at negative
#                 time so nothing needs trimming, but the coordinates
#                 collide with the text rows, which weakens the anchors
#                 and darkens the output.
#   AUDIO_MODE    "timeline" puts the pinned audio on this clip's own
#                 timeline so the model continues it. "ref" is the stock
#                 placement, which the model imitates instead: similar
#                 music, not the same recording, plus a tick at the join.
#   CROP          only ever applied when an aspect ratio changed between
#                 clips, which the resolution check now refuses outright.
ENCODE_MODE = "video"
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"
CROP = "disabled"


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]; matches the stock helper
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_tail_audio(audio_vae, audio, seconds):
    """Encode the last `seconds` of a clip's audio with the H3 audio VAE.

    Returns ([1, 32, 2, T] latent, T) where T counts 40 Hz latent steps,
    matching what the layout calls ref_audio_t.
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: context_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _steps_for_frames(n):
    """Latent steps covering exactly n pixel frames from cycle position 0.

    Returns None when no whole number of steps covers n. The video VAE's
    steps alternate 1, 4, 4, 4, 4 pixel frames, so only certain totals are
    reachable: 1, 5, 9, ... and of the windows this node offers, 5, 22, 39
    and 56 land on 2, 7, 12 and 17 steps. The 1-frame window does not,
    because the last step of a clip spans 4 frames, not 1.
    """
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _video_tail_from_latent(latent, n):
    """Slice the last n pixel frames of video straight out of a generated
    H3 latent, skipping the h264 decode and the VAE encode.

    Returns (blocks, offsets, covered) in the same shape the encode path
    produces, so everything downstream is unchanged.

    This is only sound because the tail window always starts at cycle
    position 0. A clip is 17g+5 frames, which is 5g+2 latent steps; the
    windows are 2, 7, 12 and 17 steps; and 5g+2 minus any of those is a
    multiple of 5. So the sliced run has the same 1, 4, 4, 4, 4 phase as a
    freshly encoded one and _step_offsets applies unchanged. Asserted
    below rather than assumed, because if it ever stopped holding the
    pinned content would silently disagree with the positions written for
    it and the join would land at the wrong instant.
    """
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "h3_motion_context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 5, 22, "
            "39 or 56, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "h3_motion_context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "h3_motion_context: the %d step tail of a %d step latent starts "
            "at cycle position %d, not 0, so its frame spans would not match "
            "the positions written for them. Clip lengths are meant to make "
            "this impossible; refusing rather than rendering a shifted join."
            % (steps, total, start % 5))
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "h3_motion_context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the signed fraction of a step by
    which the clip's audio grid overshoots its last pixel frame.

    H3 rounds the audio grid to the NEAREST step, not up, so overhang is
    negative for a third of legal clip lengths. 5/3 of a frame count
    lands on .0, .333 or .667 and never on .5, so there are exactly three
    cases:

        frames % 3 == 0   243 wants 405.00, allocates 405, overhang    0
        frames % 3 == 1   124 wants 206.67, allocates 207, overhang +1/3
        frames % 3 == 2   260 wants 433.33, allocates 433, overhang -1/3

    A positive overhang means the latent's final step reaches past the
    last frame, a negative one means it stops short. Either way the
    caller compensates the placement with it, so the pinned content lands
    where its samples actually sit. The decoded-audio path never sees
    this because match_tail cuts at the frame.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # unbatched [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    # legal values are exactly 0, +1/3 and -1/3; the band is the widest
    # one that admits all three and still rejects a grid that is out by a
    # whole step or more
    if not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class MiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "context_length": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Frames of the previous clip's picture to "
                               "carry over. Only these lengths are whole "
                               "numbers of latent steps, so only these are "
                               "offered. 5 is just barely fluid, 22 is "
                               "nearly seamless. Longer windows pin more "
                               "motion but come off the front of the "
                               "delivered clip, so 56 spends 2.3 seconds of "
                               "the render on frames you throw away."}),
                "audio_context_length": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Frames of tail audio to pin, independent of "
                               "the picture window. 0 follows it. The window "
                               "is END-aligned with the pinned video, so "
                               "this only controls how far back the sound "
                               "reaches. Multiples of 3 land exactly on the "
                               "40 Hz audio grid and multiples of 24 are "
                               "whole seconds: 24 pins the last second. "
                               "Off-grid values are widened to the nearest "
                               "whole step."}),
            },
            "optional": {
                "context_frames": ("IMAGE", {
                    "tooltip": "Decoded frames of the previous clip. Used "
                               "when no context_latent is wired. When one "
                               "is, the picture comes from it instead and "
                               "this is ignored."}),
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (the "
                               "same one you wire into the decode nodes). "
                               "Supplies both picture and sound, sliced "
                               "straight out, skipping the decode and "
                               "re-encode that cost a little quality at "
                               "every link of a chain. Must be the same "
                               "resolution as the clip being generated."}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Supply with context_audio to "
                               "carry the previous clip's tail sound across "
                               "the join. Not needed when context_latent is "
                               "wired."}),
                "context_audio": ("AUDIO", {
                    "tooltip": "Audio of the previous clip. The tail "
                               "matching the pinned frames is encoded and "
                               "pinned alongside them. Ignored when "
                               "context_latent is wired."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin a run of consecutive frames from a previous clip as "
                   "never-denoised conditioning rows, so the model reads real "
                   "motion instead of guessing it from a single still. With "
                   "context_latent wired, both picture and sound are sliced "
                   "from the previous clip's latent, skipping the decode and "
                   "re-encode that cost a little quality at every link. A "
                   "last_frame anchor on the incoming conditioning is kept "
                   "and pinned alongside the head.")

    def apply(self, conditioning, vae, latent, context_length,
              audio_context_length=24, context_frames=None,
              context_latent=None, audio_vae=None, context_audio=None):
        if context_latent is None and context_frames is None:
            return (conditioning, 0)
        encode_mode, anchor_mode = ENCODE_MODE, ANCHOR_MODE
        audio_mode, crop = AUDIO_MODE, CROP
        context_length = int(context_length)
        _ensure_layout_ok()

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        # Decide where the pinned VIDEO comes from before anything else,
        # because it decides how many frames are even available. Slicing it
        # out of the previous clip's latent removes an h264 decode and a
        # VAE encode from the path, and the blocks come out bit-identical
        # to what the model produced rather than a reconstruction of it.
        # A wired latent supplies the picture as well as the sound: the
        # pinned blocks are then exactly the steps the model produced,
        # with no h264 decode, no resize and no VAE round trip to shift
        # colour or contrast. Frames are the path when no latent is wired.
        if context_latent is not None:
            src_video = _video_from_latent(context_latent)
            src_w = int(src_video.shape[4]) * 16
            src_h = int(src_video.shape[3]) * 16
            if src_w != width or src_h != height:
                # a latent cannot be resized. Falling back to frames here
                # would quietly take the lossy path on a graph the user
                # thinks is fine, and a resolution change mid-chain is
                # nearly always a mistake, so say so instead.
                raise ValueError(
                    "h3_motion_context: context_latent is %dx%d but this "
                    "clip is %dx%d. A latent cannot be resized, so the "
                    "previous clip has to be regenerated at this "
                    "resolution, or the chain restarted here."
                    % (src_w, src_h, width, height))
            if int(src_video.shape[1]) != int(video.shape[1]):
                raise ValueError(
                    "h3_motion_context: context_latent has %d channels, "
                    "this clip has %d. That is not an H3 video latent from "
                    "the same model."
                    % (int(src_video.shape[1]), int(video.shape[1])))
            available = _pixel_frames(int(src_video.shape[2]))
            video_src = "latent"
        else:
            available = int(context_frames.shape[0])
            video_src = "pixels"

        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: no frames available to pin")
        if n < context_length:
            _LOG.warning("h3_motion_context: only %d frames available, pinning %d",
                         available, n)

        if encode_mode == "video":
            # snap down to the VAE grid BEFORE slicing, so the frames encoded
            # are exactly the frames the latent steps will cover (see
            # VIDEO_RUN_GRID). Slicing the last n and letting the VAE keep the
            # first `covered` of them would pin a run ending before the clip
            # does, and the join would jump by the difference.
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: %d frames is off the VAE grid; pinning "
                    "the last %d instead (usable runs: 1, 5, 22, 39, 56)", n, run)
            n = run

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        if video_src == "latent" and _steps_for_frames(n) is None:
            # every window the node offers is a whole number of steps, so
            # reaching this means the grid moved underneath us
            raise RuntimeError(
                "h3_motion_context: a %d frame window is not a whole number "
                "of latent steps. VIDEO_RUN_GRID no longer matches the "
                "VAE; refusing rather than rendering a shifted join." % n)

        if video_src == "latent":
            blocks, offsets, covered = _video_tail_from_latent(
                context_latent, n)
            span = covered
        else:
            # the LAST n frames of the incoming clip become the pinned run
            tail = _resize(context_frames[available - n:], width, height, crop)

        if video_src == "pixels" and encode_mode == "video":
            # one call; the VAE reads the batch axis as time and compresses
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: video-mode encode returned shape %s, "
                    "expected [B,C,T,H,W]. Try encode_mode=frames."
                    % (tuple(getattr(enc, "shape", ())),))
            steps = int(enc.shape[2])
            offsets = _step_offsets(steps)
            covered = _pixel_frames(steps)
            if covered != n:
                # n was snapped to the grid above, so a mismatch here means
                # the VAE's downscale formula changed underneath us and the
                # pinned content no longer lines up with the positions we
                # would write. Refuse rather than render a shifted join.
                raise RuntimeError(
                    "h3_motion_context: %d frames encoded to %d latent steps "
                    "covering %d frames; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (n, steps, covered))
            blocks = [enc[:, :, k:k + 1] for k in range(steps)]
            span = covered
        elif video_src == "pixels":
            blocks, offsets = [], []
            for i in range(n):
                blocks.append(vae.encode(tail[i:i + 1]))
                offsets.append(i)
            span = n

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            # Straight to stock. `resolved_frame_index` is the real pixel
            # frame the block sits at, which ComfyUI 0.33 accepts for any
            # value, so there is nothing to smuggle and nothing to rewrite
            # afterwards. One keyframe per latent step rather than a single
            # multi-step block: `indices` already carries each step's real
            # offset, which keeps the encode_mode=frames path (offsets
            # 0..n-1, one still each) on exactly the same code. Stock lays
            # a 1-step latent at cursor + FRAME_RESCALE * index either way,
            # so the two forms produce identical rows.
            keyframes.append({
                "resolved_frame_index": p,
                "latent": blk,
            })

        ref_audio_t = 0
        audio_ref = None
        audio_kf = None
        audio_end_frame = None
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            # the audio window is independent of the video one: audio cond
            # rows cost rows but never cost delivered frames
            a_frames = int(audio_context_length) or span
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info("h3_motion_context: both context_latent and "
                              "context_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "context_latent instead.")
                audio_latent, ref_audio_t = _encode_tail_audio(
                    audio_vae, context_audio, a_frames / float(FPS))
                overhang = 0.0  # decoded audio was match_tail-cut at the frame
                audio_src = "vae"
            if audio_mode == "timeline":
                # end-align the audio window with the pinned video: both are
                # the tail of clip A, so both must end at the same instant
                # of the new timeline, frame `span` in head mode (where
                # A's last frame sits), frame 0 in before mode. On the
                # latent path the sliced content overshoots A's last
                # frame by `overhang` of a step, signed, because H3
                # rounds its audio grid to the nearest step and so falls
                # short as often as it reaches past. The end coordinate
                # moves by exactly that much, and a keyframe index is a
                # plain multiplier so it takes a fractional frame.
                end_frame = float(span if anchor_mode == "head" else 0)
                end_frame += overhang / FRAME_RESCALE
                # then snap the window onto the target's own audio grid.
                # The end coordinate is FRAME_RESCALE * end_frame, and
                # FRAME_RESCALE is 5/3, so unless that product happens to
                # be a whole number the pinned rows land between the
                # integer coordinates the target's audio rows occupy. A
                # third of a step is 8.3 ms, which is the size of the
                # constant late offset measured on chained clips. Whether
                # it lands on or off the grid depends on the window
                # length, the path, and the clip's own grid overhang, so
                # it cycles rather than staying put. Rounding the end
                # coordinate to the nearest integer costs at most a third
                # of a step of placement and puts the pinned content on
                # the same grid as the sound being generated from it.
                end_coord = round(FRAME_RESCALE * end_frame)
                end_frame = end_coord / FRAME_RESCALE
                # Stock places a keyframe's audio window STARTING at
                # FRAME_RESCALE * index past the target origin and running
                # forward. We need it to END at the join, so the index is
                # the start of a window `ref_audio_t` steps wide:
                #
                #   start coord = FRAME_RESCALE * end_frame - ref_audio_t
                #   index       = end_frame - ref_audio_t / FRAME_RESCALE
                #
                # which is fractional, and negative whenever the window is
                # longer than the pinned head, which it normally is. Both
                # are legal arithmetic in the layout and neither is
                # reachable through the stock Add Guide node, so
                # layout_contract checks them before the first render.
                audio_kf = {
                    "resolved_frame_index": (end_frame
                                             - ref_audio_t / FRAME_RESCALE),
                    "audio_latent": audio_latent,
                }
                audio_end_frame = end_frame
            else:
                # stock reference placement: the window sits in its own
                # span ahead of the target, which is what makes the model
                # imitate the sound rather than continue it. Kept as the
                # comparison the timeline mode is measured against.
                audio_ref = {
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                }

        # MERGE with any keyframes already on the conditioning instead of
        # replacing them. A last_frame anchor from the upstream node, or an
        # Add Guide anchor, is a legitimate companion to a chained head: the
        # pinned run decides how the clip starts, the anchor decides where
        # it ends. They need no special handling now that every keyframe
        # carries its real index, ours included, and stock compensates all
        # of them for references the same way.
        #
        # Anchors inside the pinned head are dropped: the pinned run
        # already decides those frames, and a second cond block at the
        # same coordinate would fight it.
        head_end = span if anchor_mode == "head" else 0
        tail_kfs = [audio_kf] if audio_kf is not None else []
        out = []
        dropped = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            kept = []
            for kf in prior:
                p = kf.get("resolved_frame_index", 0)
                if p >= frame_count:
                    raise ValueError(
                        "h3_motion_context: the conditioning carries a "
                        "keyframe anchored at frame %s, but this clip is "
                        "only %d frames. Wire the conditioning and the "
                        "latent from the same node." % (p, frame_count))
                if p < head_end:
                    dropped.append(p)
                    continue
                kept.append(dict(kf))
            d["minimax_keyframes"] = kept + keyframes + tail_kfs
            out.append([emb, d])
        if dropped:
            _LOG.warning(
                "h3_motion_context: dropped %d keyframe anchor(s) at "
                "frame(s) %s: the pinned head already decides frames "
                "0..%d. A last_frame anchor is kept.",
                len(dropped), sorted(set(dropped)), head_end - 1)

        if audio_ref is not None:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        _LOG.info("h3_motion_context: video from %s, %s/%s, %d frames -> %d "
                  "cond blocks at indices %d..%d, %d frame clip at %dx%d, "
                  "trim %d, audio %s",
                  video_src, encode_mode, anchor_mode, n, len(blocks),
                  indices[0], indices[-1], frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
                      "on the timeline ending at frame %.3f" % audio_end_frame
                      if audio_end_frame is not None
                      else "stock ref placement"))
                  if ref_audio_t else "off")
        return (out, trim)


class MiniMaxH3MotionContextTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3,
    so the grid rarely lands on a frame boundary. It rounds to the
    NEAREST step, which means a clip ships either about 8.3 ms more sound
    than picture or about 8.3 ms less, depending on its length:

        frames % 3 == 0   243 wants 405.00 steps, gets 405, exact
        frames % 3 == 1   124 wants 206.67 steps, gets 207, sound is long
        frames % 3 == 2   260 wants 433.33 steps, gets 433, sound is short

    Either way the error compounds. Concatenate two clips and the second
    seam is out by 16.7 ms, three and it is 25 ms, and it grows without
    bound down a chain. It reads as a faint dampening at the first join
    and a short click at later ones. Matching the tail to exactly
    frames/fps stops it accumulating: a long tail is truncated, a short
    one is zero-padded. The padded samples are sound the model never
    generated, so silence is the only honest fill.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make the audio duration equal frames/fps "
                               "exactly, trimming a long tail or padding a "
                               "short one with silence. H3 rounds its audio "
                               "grid to the nearest step, so each clip "
                               "carries about 8ms too much or too little "
                               "sound, which accumulates at every join in a "
                               "chain."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Remove the leading pinned frames from a decoded H3 clip, "
                   "trimming picture and sound by the same duration.")

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        out_images = images[n:] if n else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_motion_context: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    # H3 rounds to the nearest audio step, so a third of
                    # clip lengths ship slightly LESS sound than picture
                    # rather than more. The missing samples are sound
                    # that was never generated, so zero is the honest
                    # fill; anything else would fabricate or attenuate
                    # real content to hide a seam. Leaving it short
                    # instead drifts every later clip earlier, and unlike
                    # the long case that error compounds down the chain.
                    # This also restores what the vae path assumes when
                    # it sets overhang to 0.
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform,
                                                       (0, missing))
                    _LOG.info("h3_motion_context: tail padded %d zero "
                              "samples (%.2fms) so audio matches %d "
                              "frames exactly",
                              missing, missing / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_motion_context: trimmed %d leading frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will run %.3fs ahead of the picture.",
                      n, total - n, n / float(fps))

        return (out_images, out_audio)


def _under_output(path):
    """Resolve `path` inside ComfyUI's output folder. None if it would escape.

    Absolute paths are allowed only when they already live under output/.
    Relative paths join onto output/. This is what keeps latent_path from
    becoming an arbitrary file read or a delete-anywhere on Clear.
    """
    root = os.path.realpath(folder_paths.get_output_directory())
    p = (path or "").strip().strip('"').strip("'") or "h3_context"
    resolved = os.path.realpath(p if os.path.isabs(p) else os.path.join(root, p))
    if resolved != root and not resolved.startswith(root + os.sep):
        return None
    return resolved


def _resolve_latent_path(path, clip_index=0):
    """Turn the loader's path input into a concrete file under output/.

    Accepts a path relative to ComfyUI's output folder, an absolute path
    that already lives there, or a directory in either form. For a
    directory, clip_index must be a positive slot: clip 1 is
    *_00001.safetensors. Auto-mode files carry a trailing underscore
    (*_00001_.safetensors) and are never matched, because their numbers
    count runs and could hold a reject. clip_index 0 is handled by the
    Load node itself (no file, first clip).
    """
    c = _under_output(path)
    if not c:
        raise FileNotFoundError(
            "h3_motion_context: path must stay inside the ComfyUI output folder.")
    if os.path.isfile(c):
        return c
    if os.path.isdir(c):
        idx = int(clip_index)
        if idx <= 0:
            raise FileNotFoundError(
                "h3_motion_context: clip_index 0 does not load a file.")
        # indexed slots use the natural name: clip 2 lives in
        # *_00002.safetensors. Auto-mode files carry a trailing
        # underscore (*_00002_.safetensors) and are deliberately
        # NOT matched: their numbers count runs, not clips, so a
        # reject could be sitting in any of them.
        endings = ("_%05d.safetensors" % idx,
                   "_clip%03d.safetensors" % idx)  # older versions
        files = [os.path.join(c, f) for f in os.listdir(c)
                 if f.endswith(endings)]
        if not files:
            near = [f for f in os.listdir(c)
                    if f.endswith("_%05d_.safetensors" % idx)]
            hint = ""
            if near:
                hint = (" Found %s, which is an auto-numbered save "
                        "(trailing underscore = numbered by RUN, so "
                        "it may be a reject). If it really is clip "
                        "%d, rename it to drop the trailing "
                        "underscore: %s" %
                        (near[0], idx,
                         near[0].replace("_%05d_" % idx,
                                         "_%05d" % idx)))
            raise FileNotFoundError(
                "h3_motion_context: no saved latent for clip %d "
                "(no *_%05d.safetensors in %s).%s"
                % (idx, idx, c, hint))
        return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a folder under the "
        "ComfyUI output directory." % ((path or "").strip() or "h3_context"))


def _latent_folder(path):
    """Directory that holds this chain's slot files, or None.

    Refuses anything outside ComfyUI's output folder so Clear cannot
    delete numbered safetensors elsewhere on disk.
    """
    c = _under_output(path)
    if not c:
        return None
    if os.path.isdir(c):
        return c
    if os.path.isfile(c):
        return os.path.dirname(c)
    return None


def _clip_slot_exists(latent_path, clip_index=1):
    try:
        _resolve_latent_path(latent_path, int(clip_index))
        return True
    except FileNotFoundError:
        return False


_CHAIN_SLOT_FILE = re.compile(
    r"(?:_\d{5}|_\d{5}_|_clip\d{3})\.safetensors(?:\.tmp)?$"
)


def _clear_clip_slots(latent_path):
    """Delete numbered chain slots. Custom filenames are left alone."""
    folder = _latent_folder(latent_path)
    if not folder:
        return 0
    removed = 0
    for name in os.listdir(folder):
        if not _CHAIN_SLOT_FILE.search(name):
            continue
        fp = os.path.join(folder, name)
        try:
            os.remove(fp)
        except OSError:
            gc.collect()
            os.remove(fp)
        removed += 1
    if removed:
        _LOG.info("h3_motion_context: cleared %d chain slot(s) from %s",
                  removed, folder)
    return removed


# --- Boyo fork: approve-gated video saving ---------------------------------
#
# Cache for the most recently produced VHS Combine output path. Named
# obscurely and namespaced under a Boyo-specific prefix rather than
# something generic like _last_clip, so a second H3-chaining fork sharing
# this ComfyUI process can't collide with it by reaching for the same
# module attribute name. Holds nothing sensitive, just a filesystem path.
_BOYO_H3MC_VHSCACHE_7f2a9d = {"path": None}


def _boyo_clip_slot_path(folder_path, clip_index, ext):
    """Numbered slot for an approved clip: clip_00003.mp4.

    Matches the naming convention MiniMaxH3MotionContextSaveLatent already
    uses for its safetensors slots, so an approved video and its matching
    latent share the same clip_index across both save mechanisms.
    """
    folder = _under_output(folder_path)
    if not folder:
        raise FileNotFoundError(
            "h3_motion_context: folder must stay inside the ComfyUI "
            "output folder.")
    os.makedirs(folder, exist_ok=True)
    idx = int(clip_index)
    if idx <= 0:
        raise ValueError(
            "h3_motion_context: clip_index must be >= 1 to save an "
            "approved clip.")
    return os.path.join(folder, "clip_%05d%s" % (idx, ext))


_BOYO_APPROVED_CLIP_FILE = re.compile(r"^clip_(\d{5})(\.[A-Za-z0-9]+)$")

# Fixed location, not a widget -- keeps this in one predictable place
# alongside the latent slots rather than one more folder to configure.
_BOYO_FRAME_SUBFOLDER = "h3_context/frames"


def _boyo_frame_slot_path(clip_index):
    """Numbered slot for an approved last-frame PNG: frame_00003.png.

    Same numbering as MiniMaxH3MotionContextSaveLatent's clip_index, kept
    in a fixed subfolder so it never needs its own configuration.
    """
    idx = int(clip_index)
    if idx <= 0:
        raise ValueError(
            "h3_motion_context: clip_index must be >= 1 for an approved "
            "frame slot.")
    folder = _under_output(_BOYO_FRAME_SUBFOLDER)
    if not folder:
        raise FileNotFoundError(
            "h3_motion_context: frame folder must stay inside the "
            "ComfyUI output folder.")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "frame_%05d.png" % idx)


def _boyo_list_output_subfolders():
    """Immediate subfolders of the ComfyUI output directory, sorted
    alphabetically, for the Folder Concatenate node's folder dropdown.

    This list is captured whenever ComfyUI (re)builds this node's
    definition -- on startup, or a manual "reload custom nodes" / canvas
    refresh -- not on every graph execution. A folder created after that
    point will not appear in the dropdown until the next reload. Restart
    ComfyUI or refresh node definitions if a folder you just created
    with H3 Approve Save is missing from the list.
    """
    try:
        root = folder_paths.get_output_directory()
        names = sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    except OSError:
        names = []
    return names or ["(no folders found in output/ yet)"]


def _boyo_approved_clips(folder):
    """Numbered clip files directly inside `folder`, in clip order.

    Returns a list of (clip_index, path) tuples. Raises ValueError if
    the folder contains more than one file extension among matches --
    ffmpeg's concat demuxer stream-copies without re-encoding, which
    only works when every input shares the same codec and container,
    and a mixed folder is the most likely way that assumption breaks.
    """
    entries = []
    exts = set()
    for name in os.listdir(folder):
        m = _BOYO_APPROVED_CLIP_FILE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        ext = m.group(2).lower()
        exts.add(ext)
        entries.append((idx, os.path.join(folder, name)))
    if not entries:
        raise ValueError(
            "h3_motion_context: no approved clips (clip_00001.mp4 and so "
            "on) found in %s. Approve or Chain some clips into this "
            "folder first." % folder)
    if len(exts) > 1:
        raise ValueError(
            "h3_motion_context: folder has mixed clip formats (%s). "
            "Stream-copy concatenation needs every clip to share the "
            "same codec and container. Re-save this chain's clips with "
            "one consistent format." % ", ".join(sorted(exts)))
    entries.sort(key=lambda t: t[0])
    return entries


def register_chain_routes():
    """POST routes for the Chain node's start, Clear, and approve-save."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(register_chain_routes, "_done", False):
        return

    from .csrf_guard import require_same_origin

    @server.routes.post("/h3_motion_context/slot_exists")
    @require_same_origin
    async def _slot_exists_route(request):
        data = await request.json()
        exists = _clip_slot_exists(data.get("latent_path") or "h3_context",
                                   int(data.get("clip_index") or 1))
        return web.json_response({"exists": exists})

    @server.routes.post("/h3_motion_context/clear_latents")
    @require_same_origin
    async def _clear_latents_route(request):
        data = await request.json()
        n = _clear_clip_slots(data.get("latent_path") or "h3_context")
        return web.json_response({"removed": n})

    # Boyo fork: promote VHS Combine's cached output into a numbered chain
    # slot. Route path is namespaced under boyonodes_h3mc/ rather than
    # h3_motion_context/ so it can't collide with another fork's routes
    # sharing this same ComfyUI process.
    @server.routes.post("/boyonodes_h3mc/approve_save")
    @require_same_origin
    async def _boyo_approve_save_route(request):
        data = await request.json()
        src = _BOYO_H3MC_VHSCACHE_7f2a9d.get("path")
        if not src:
            _LOG.warning("boyo-h3mc: approve_save called with no cached "
                         "VHS output yet")
            return web.json_response(
                {"ok": False, "error": "no cached VHS output yet"},
                status=409)
        if not os.path.isfile(src):
            _LOG.warning("boyo-h3mc: cached VHS output no longer on disk: %s",
                        src)
            return web.json_response(
                {"ok": False, "error": "cached VHS output no longer on "
                                       "disk: %s" % src},
                status=409)
        try:
            dest = _boyo_clip_slot_path(
                data.get("folder") or "h3_context/approved",
                data.get("clip_index") or 0,
                os.path.splitext(src)[1] or ".mp4")
        except (FileNotFoundError, ValueError) as exc:
            _LOG.warning("boyo-h3mc: approve_save rejected: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)},
                                     status=400)
        shutil.copy2(src, dest)
        _LOG.info("boyo-h3mc: approved clip %s saved to %s (from %s)",
                  data.get("clip_index"), dest, src)
        return web.json_response({"ok": True, "path": dest})

    register_chain_routes._done = True


def _write_safetensors(path, tensors):
    # safetensors load_file memory-maps on Windows. Overwriting a mapped
    # slot (re-roll, or Load 0 aimed at the file Save is about to replace)
    # fails with os error 1224. Write a sibling temp file and replace.
    tmp = path + ".tmp"
    try:
        _st_save(tensors, tmp,
                 metadata={"format": "h3_motion_context_av_v1"})
        try:
            os.replace(tmp, path)
        except OSError:
            gc.collect()
            os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


class MiniMaxH3MotionContextSaveLatent:
    """Save an H3 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run
    -- so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent can't serialise H3's nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The sampler's output latent (the same one "
                               "you wire into the decode nodes)."}),
                "filename_prefix": ("STRING", {
                    "default": "h3_context/clip",
                    "tooltip": "Saved under the ComfyUI output folder. The "
                               "default keeps all chain latents in one "
                               "folder so the Load node can always pick "
                               "the newest."}),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Which clip of the chain THIS is. Saves to "
                               "that clip's fixed slot, so a re-roll "
                               "overwrites its own reject instead of "
                               "stacking new files. First clip: 1 here "
                               "and 0 on the Load node. Clip 2: 2 here "
                               "and 1 on the Load node. 0 = old behaviour, "
                               "a new numbered file every run (numbers "
                               "count runs, not clips)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Save the sampler's AV latent so the next run's Motion "
                   "Context node can pin audio from it via the matching "
                   "Load node.")

    def save(self, latent, filename_prefix, clip_index=0):
        if _st_save is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot save latents.")
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: latent has no audio stream; wire the "
                "sampler output of an H3 AV graph.")
        video = parts[0].cpu().contiguous()
        audio = parts[1].cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            # fixed slot with the natural name: clip 2 -> *_00002. A
            # re-roll of this clip overwrites its own save, so rejects
            # never accumulate or get loaded later. Auto mode (below)
            # keeps a trailing underscore, which is what excludes its
            # run-numbered files from indexed loading.
            path = os.path.join(folder, "%s_%05d.safetensors"
                                % (filename, int(clip_index)))
        else:
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, counter))
        _write_safetensors(path, {"video": video, "audio": audio})
        _LOG.info("h3_motion_context: saved AV latent to %s (video %s, "
                  "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        return (path,)


class MiniMaxH3MotionContextLoadLatent:
    """Load a saved H3 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. Generating clip 2 from
    clip 1: Load node 1, Save node 2. Use H3 Motion Context Chain to
    advance the pair and queue: Approve walks forward, Run/Re-roll stays
    on the current slot (use it instead of ComfyUI's Run), Chain is
    Approve on a loop (at 0/1 with no clip 1 yet it generates that
    first), Reset sets Load 0 / Save 1, Clear latents deletes numbered
    slots. All three nodes must sit in the same canvas group.

    At 0 there is no previous clip: the loader returns nothing and Motion
    Context passes the conditioning through. First clip of a chain is
    Load 0 / Save 1, then Load 1 / Save 2, and so on.

    The output is ONLY for the Motion Context node's context_latent input.
    It is not a decodable latent -- do not wire it into VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "h3_context",
                    "tooltip": "A saved latent file, or a folder (relative "
                               "paths resolve against the ComfyUI output "
                               "directory). Pointing at a specific FILE "
                               "always loads that file when clip_index "
                               "is greater than 0, ignoring the index. "
                               "clip_index 0 never reads a file."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "The clip to CONTINUE FROM: that clip's "
                               "slot is loaded. First clip: 0 here and 1 "
                               "on the Save node (nothing is loaded). "
                               "Clip 2 from clip 1: 1 here and 2 on the "
                               "Save node."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Load a latent saved by H3 Motion Context Save Latent, "
                   "for the context_latent input only.")

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0):
        # the path string stays constant while the file behind it changes
        # (an overwritten slot), so cache on the resolved file identity
        # instead -- otherwise ComfyUI would happily serve a stale latent
        # forever. Index 0 never reads a file.
        if int(clip_index) <= 0:
            return "disabled"
        try:
            p = _resolve_latent_path(latent_path, clip_index)
            return "%s:%d:owned" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, latent_path, clip_index=0):
        if int(clip_index) <= 0:
            return (None,)
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot load latents.")
        path = _resolve_latent_path(latent_path, clip_index)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        # copy off the mmap so a later Save can overwrite this slot on Windows
        video = data.pop("video").contiguous().clone()
        audio = data.pop("audio").contiguous().clone()
        _LOG.info("h3_motion_context: loaded AV latent from %s", path)
        # a plain list, not a NestedTensor: only this repo's context_latent
        # input accepts it, which is the point -- it cannot be mistaken
        # for a decodable latent without failing loudly downstream
        return ({"samples": [video, audio]},)


class MiniMaxH3MotionContextChain:
    """Approve, Run/Re-roll, auto-chain, reset indices, or clear slots.

    Load, Save, and this node must sit in the same canvas group or the
    buttons do nothing. Chain is Approve on a loop. At Load 0 / Save 1
    with no clip 1 on disk it generates that first clip instead of
    advancing into a missing file. segments > 0 stops Chain after that
    many clips; 0 keeps going until Stop. Reset only sets 0/1. Clear
    latents deletes numbered chain slots, not custom filenames.
    Execute is a no-op; the buttons and two HTTP routes do the work.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "How many clips Chain generates before it "
                               "stops. 0 keeps going until you click Stop."
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Approve advances Load/Save and runs the next clip. "
                   "Run/Re-roll queues the current slot (use this instead "
                   "of ComfyUI's Run). Chain is Approve on a loop; at "
                   "Load 0 / Save 1 with no clip 1 saved it generates "
                   "clip 1 first. segments is how many clips Chain runs "
                   "before stopping (0 = until Stop). Reset sets Load 0 / "
                   "Save 1. Clear latents deletes numbered chain slots. "
                   "Load, Save, and this node must sit in the same canvas "
                   "group or the buttons do nothing.")

    def noop(self, segments=0):
        return ()


class BoyoH3ApproveSave:
    """Cache VHS Combine's output path for the Chain node to promote.

    No disk I/O happens in this node. VHS Combine has already produced a
    file at fixed, hardwired encode settings -- our only job is
    remembering where it landed so the Chain node's Approve/Chain click
    can copy it into a numbered chain slot BEFORE Load/Save indices
    advance and the next clip is queued. That ordering is what guarantees
    the clip promoted is the one you just reviewed, not whatever the next
    render produces.

    Wire this after VHS Combine (save_output can stay off -- this uses
    whatever VHS wrote to its temp folder). Runs on every execution,
    including rejected Re-rolls: caching a path costs nothing, and
    nothing reaches the output folder unless Approve or Chain is clicked
    on the Chain node.

    OUTPUT_NODE = True is required here even though this is a pure
    passthrough: its `filenames` output isn't wired onward to anything,
    and ComfyUI prunes nodes that don't feed an OUTPUT_NODE from the
    execution graph. Without this flag the node is silently skipped and
    the cache below is never populated, which surfaces two steps
    downstream as "no cached VHS output yet" on Approve -- a confusing
    place to discover that this node never ran at all.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES", {
                    "tooltip": "Wire from VHS Combine's Filenames output. "
                               "save_output can stay off. Per VHS's own "
                               "docs the LAST path in the list is its "
                               "most complete output (audio-muxed if this "
                               "clip has sound), which is what gets "
                               "promoted on Approve/Chain."}),
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("filenames",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "BoyoNodes/H3"
    DESCRIPTION = ("Remembers VHS Combine's output path for the Chain "
                   "node's Approve/Chain buttons to promote into a "
                   "numbered chain slot. Passthrough only -- no file I/O "
                   "happens here.")

    def execute(self, filenames):
        try:
            _saved_flag, paths = filenames
        except (TypeError, ValueError):
            raise ValueError(
                "h3_motion_context: filenames is not a VHS_FILENAMES "
                "tuple. Wire this from VHS Combine's Filenames output.")
        if not paths:
            raise ValueError(
                "h3_motion_context: VHS Combine reported no output files.")
        # per VHS's own docs, the LAST path is its most complete output --
        # not matched by name, since that naming is VHS's implementation
        # detail and could change
        _BOYO_H3MC_VHSCACHE_7f2a9d["path"] = paths[-1]
        _LOG.info("boyo-h3mc: cached VHS output -> %s", paths[-1])
        return (filenames,)


class BoyoH3FolderConcatenate:
    """Stitch a folder of approved H3 clips into one file, in clip order.

    Reads clip_00001.mp4, clip_00002.mp4 and so on straight off disk --
    the same numbered slots H3 Approve Save writes into -- and joins
    them with ffmpeg's concat demuxer using a stream copy: no decode, no
    re-encode, no clips loaded into ComfyUI at all. That matters at
    scale: loading twenty approved clips onto the canvas as IMAGE/AUDIO
    inputs to join them would mean twenty decoded videos in memory at
    once, which is exactly the crash this node exists to avoid. Its
    only inputs are its own widgets; there is nothing to wire.

    Stream-copy concatenation needs every input clip to share the same
    codec and container, which holds automatically here because every
    approved clip came out of the same hardwired VHS Combine settings.
    No crossfade is applied or needed -- continuity across each join
    already lives in the latent handoff from the chaining nodes, not in
    the edit.

    Requires ffmpeg on PATH.

    Typical use: build a chain with the rest of the workflow live, then
    mute or bypass everything except this node and hit ComfyUI's plain
    Run (not Run/Re-roll -- this node needs no queue management, it is
    a pure filesystem job), then mute this node again before generating
    the next chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (_boyo_list_output_subfolders(), {
                    "tooltip": "A folder directly under the ComfyUI "
                               "output directory containing approved "
                               "clips (clip_00001.mp4 and so on), such "
                               "as one you pointed H3 Approve Save at. "
                               "This list is captured when ComfyUI "
                               "builds this node's definition, not live "
                               "-- restart ComfyUI or refresh node "
                               "definitions if a folder you just "
                               "created is missing from it."}),
                "output_filename": ("STRING", {
                    "default": "joined",
                    "tooltip": "Filename for the joined result (no "
                               "extension needed -- it matches the "
                               "approved clips' own container). Saved "
                               "into a 'joined' subfolder of the folder "
                               "selected above, so approved clips and "
                               "their joins stay separated."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_path",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "BoyoNodes/H3"
    DESCRIPTION = ("Joins every approved clip in the selected folder "
                   "into one file with an ffmpeg stream-copy concat -- "
                   "no re-encode, no clips loaded onto the canvas. "
                   "Saves into a 'joined' subfolder alongside the "
                   "approved clips. Requires ffmpeg on PATH.")

    def execute(self, folder, output_filename):
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "h3_motion_context: ffmpeg was not found on PATH. This "
                "node shells out to ffmpeg for a stream-copy concat; "
                "install it and make sure it's on PATH.")
        resolved = _under_output(folder)
        if not resolved or not os.path.isdir(resolved):
            raise ValueError(
                "h3_motion_context: %r is not a folder under the ComfyUI "
                "output directory. Pick a real folder from the dropdown "
                "-- if you just created one, refresh node definitions "
                "or restart ComfyUI so it appears in the list." % folder)

        clips = _boyo_approved_clips(resolved)
        ext = os.path.splitext(clips[0][1])[1]
        _LOG.info("boyo-h3mc: concatenating %d clip(s) from %s: %s",
                  len(clips), resolved,
                  ", ".join(os.path.basename(p) for _, p in clips))

        dest_dir = os.path.join(resolved, "joined")
        os.makedirs(dest_dir, exist_ok=True)
        name = (output_filename or "joined").strip() or "joined"
        dest_path = os.path.join(dest_dir, name + ext)

        # ffmpeg's concat demuxer wants a manifest file, one input per
        # line, single-quoted with embedded quotes escaped per its OWN
        # escaping rule -- this is not shell quoting, it is ffmpeg's
        # demuxer syntax, and the two are easy to conflate.
        list_fd, list_path = tempfile.mkstemp(
            suffix=".txt", prefix="boyo_h3mc_concat_")
        try:
            with os.fdopen(list_fd, "w", encoding="utf-8") as f:
                for _, path in clips:
                    escaped = path.replace("'", "'\\''")
                    f.write("file '%s'\n" % escaped)

            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", list_path, "-c", "copy", dest_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                raise RuntimeError(
                    "h3_motion_context: ffmpeg concat failed (exit %d). "
                    "A stream copy needs every clip to share IDENTICAL "
                    "codec parameters, not just the same extension -- "
                    "if these clips came from different resolutions or "
                    "encode settings, that is the most likely cause. "
                    "ffmpeg's last output:\n%s"
                    % (proc.returncode, tail))
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass

        _LOG.info("boyo-h3mc: joined %d clip(s) -> %s", len(clips), dest_path)
        return {"ui": {"text": [
            "Joined %d clips -> %s" % (len(clips), dest_path)]},
            "result": (dest_path,)}


class BoyoH3SaveApprovedFrame:
    """Save a clip's last decoded frame to its numbered slot, unconditionally.

    Wire this from wherever you already pull the last frame for manual
    FFLF feedback -- e.g. VHS Combine's images output fed into a Get
    Image from Batch node with batch_index -1. Saves on EVERY execution,
    whether or not the clip goes on to be approved, exactly like H3
    Motion Context Save Latent does for the latent: nothing reads a slot
    until the Chain node's Approve/Chain buttons advance the Load index
    into it, so a rejected clip's frame just sits there, overwritten by
    the next attempt at that same slot. Gating lives entirely on the
    LOAD side.

    clip_index is kept in sync with H3 Motion Context Save Latent's own
    clip_index automatically by the Chain node's buttons -- there is
    nothing to set by hand, and any manual edit is overwritten before
    the next render anyway.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "This clip's last decoded frame, e.g. "
                               "from Get Image from Batch (batch_index "
                               "-1) off VHS Combine's images output."}),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Which clip THIS frame belongs to. Kept "
                               "in sync with the Save Latent node's "
                               "clip_index by the Chain node's buttons. "
                               "0 saves nothing -- there is no previous "
                               "clip yet."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "BoyoNodes/H3"
    DESCRIPTION = ("Saves this clip's last frame to its numbered slot on "
                   "every render, gated only by whether Approve/Chain "
                   "ever advances a Load index far enough to read it.")

    def save(self, image, clip_index=0):
        idx = int(clip_index)
        if idx <= 0:
            _LOG.info("boyo-h3mc: SaveApprovedFrame skipped, clip_index "
                      "0 (first clip has no previous frame to save)")
            return ()
        path = _boyo_frame_slot_path(idx)
        img = image[-1] if image.ndim == 4 else image
        arr = (img.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(path)
        _LOG.info("boyo-h3mc: saved approved-frame candidate for clip "
                  "%d -> %s", idx, path)
        return ()


class BoyoH3LoadApprovedFrame:
    """Load the previous clip's approved last frame, or the real start frame.

    At clip_index 0 (no previous clip yet) this passes initial_frame
    through unchanged. Otherwise it loads the PNG H3 Save Approved Frame
    wrote for that clip -- which is only ever a frame from a clip that
    was actually approved, since Load's index can never reach a slot
    that Approve/Chain never advanced into.

    clip_index is kept in sync with H3 Motion Context Load Latent's own
    clip_index automatically by the Chain node's buttons.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_frame": ("IMAGE", {
                    "tooltip": "Your real starting image. Used only "
                               "when clip_index is 0."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "The clip to load the approved last "
                               "frame FROM. Kept in sync with the Load "
                               "Latent node's clip_index by the Chain "
                               "node's buttons. 0 passes initial_frame "
                               "through unchanged."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "BoyoNodes/H3"
    DESCRIPTION = ("Loads the approved last frame of the clip to "
                   "continue from, falling back to initial_frame when "
                   "there is no previous clip yet.")

    @classmethod
    def IS_CHANGED(cls, initial_frame, clip_index=0):
        # same reasoning as MiniMaxH3MotionContextLoadLatent's
        # IS_CHANGED: the path string is constant while the file behind
        # it changes on every re-save of that slot, so cache on the
        # resolved file's own mtime instead of the path.
        if int(clip_index) <= 0:
            return "disabled"
        try:
            p = _boyo_frame_slot_path(clip_index)
            if not os.path.isfile(p):
                return float("NaN")
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")

    def load(self, initial_frame, clip_index=0):
        idx = int(clip_index)
        if idx <= 0:
            return (initial_frame,)
        path = _boyo_frame_slot_path(idx)
        if not os.path.isfile(path):
            _LOG.warning(
                "boyo-h3mc: no approved frame for clip %d at %s; "
                "falling back to initial_frame", idx, path)
            return (initial_frame,)
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        _LOG.info("boyo-h3mc: loaded approved frame for clip %d from %s",
                  idx, path)
        return (tensor,)


class BoyoH3PromptSelect:
    """Pick one pre-drafted prompt chunk by clip_index.

    Write every clip's prompt in one text box, each chunk separated by
    a line containing only ---. clip_index selects which chunk comes
    out, using the exact same numbering as the Save Latent node's
    clip_index -- kept in sync automatically by the Chain node's
    buttons, so prompt N is always paired with generation N, a Re-roll
    reuses the same prompt untouched, and a rejected clip's prompt is
    never skipped past.

    No LLM runs here: this is the deterministic half of prompting a
    chain. Feed it whatever text an LLM (or you) already drafted --
    wire a fixed text node in for repeatable testing so an LLM's own
    variability isn't a second unknown stacked on top of this one.

    Requesting a chunk past the end reuses the LAST chunk instead of
    erroring, so an unattended Chain run doesn't die mid-sequence over
    a prompt-count mismatch. A warning is logged when that happens.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One prompt per clip. Separate chunks "
                               "with a line containing only ---."}),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Which chunk to output. Kept in sync "
                               "with the Save Latent node's clip_index "
                               "by the Chain node's buttons."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "info")
    FUNCTION = "select"
    CATEGORY = "BoyoNodes/H3"
    DESCRIPTION = ("Selects one pre-drafted prompt chunk by clip_index "
                   "from a --- separated text block.")

    @staticmethod
    def _split(text):
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        chunks, current = [], []
        for line in lines:
            if line.strip() == "---":
                chunks.append("\n".join(current).strip())
                current = []
            else:
                current.append(line)
        chunks.append("\n".join(current).strip())
        return [c for c in chunks if c]

    def select(self, text, clip_index=1):
        chunks = self._split(text)
        if not chunks:
            raise ValueError(
                "h3_motion_context: no prompt chunks found. Separate "
                "each clip's prompt with a line containing only ---.")
        idx0 = max(0, int(clip_index) - 1)
        if idx0 >= len(chunks):
            _LOG.warning(
                "boyo-h3mc: clip_index %d has no matching prompt chunk "
                "(%d written); reusing the last chunk.",
                clip_index, len(chunks))
            idx0 = len(chunks) - 1
        prompt = chunks[idx0]
        info = "chunk %d/%d" % (idx0 + 1, len(chunks))
        _LOG.info("boyo-h3mc: prompt select clip %d -> %s", clip_index, info)
        return (prompt, info)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContext": MiniMaxH3MotionContext,
    "MiniMaxH3MotionContextTrim": MiniMaxH3MotionContextTrim,
    "MiniMaxH3MotionContextSaveLatent": MiniMaxH3MotionContextSaveLatent,
    "MiniMaxH3MotionContextLoadLatent": MiniMaxH3MotionContextLoadLatent,
    "MiniMaxH3MotionContextChain": MiniMaxH3MotionContextChain,
    "BoyoH3ApproveSave": BoyoH3ApproveSave,
    "BoyoH3FolderConcatenate": BoyoH3FolderConcatenate,
    "BoyoH3SaveApprovedFrame": BoyoH3SaveApprovedFrame,
    "BoyoH3LoadApprovedFrame": BoyoH3LoadApprovedFrame,
    "BoyoH3PromptSelect": BoyoH3PromptSelect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContext": "H3 Motion Context",
    "MiniMaxH3MotionContextTrim": "H3 Motion Context Trim",
    "MiniMaxH3MotionContextSaveLatent": "H3 Motion Context Save Latent",
    "MiniMaxH3MotionContextLoadLatent": "H3 Motion Context Load Latent",
    "MiniMaxH3MotionContextChain": "H3 Motion Context Chain",
    "BoyoH3ApproveSave": "H3 Approve Save (Boyo)",
    "BoyoH3FolderConcatenate": "H3 Folder Concatenate (Boyo)",
    "BoyoH3SaveApprovedFrame": "H3 Save Approved Frame (Boyo)",
    "BoyoH3LoadApprovedFrame": "H3 Load Approved Frame (Boyo)",
    "BoyoH3PromptSelect": "H3 Prompt Select (Boyo)",
}
