import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs


PLUGIN_PATH = os.environ.get("BM3DCUDA_PLUGIN")
if not PLUGIN_PATH:
    pytest.skip(
        "set BM3DCUDA_PLUGIN to a static CUDA plugin build",
        allow_module_level=True,
    )

core = vs.core
core.std.LoadPlugin(PLUGIN_PATH)
bm3dcuda = core.bm3dcuda


def patterned_clip(format_id: int, length: int = 11) -> vs.VideoNode:
    blank = core.std.BlankClip(
        width=32, height=24, format=format_id, length=length
    )

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        output = f.copy()
        for plane in range(output.format.num_planes):
            array = np.asarray(output[plane])
            y, x = np.indices(array.shape)
            array[:] = (
                -0.1 * plane
                + ((x * (3 + plane) + y * (5 + plane) + n * (7 + plane)) % 83)
                / 255.0
            )
        output.props["TestFrame"] = n
        return output

    return core.std.ModifyFrame(blank, blank, fill)


def assert_frames_equal(left: vs.VideoNode, right: vs.VideoNode, order) -> None:
    for n in order:
        left_frame = left.get_frame(n)
        right_frame = right.get_frame(n)
        assert right_frame.props["TestFrame"] == n
        for plane in range(left_frame.format.num_planes):
            assert np.array_equal(
                np.asarray(left_frame[plane]), np.asarray(right_frame[plane])
            ), (n, plane)


@pytest.mark.parametrize(
    "arguments",
    [
        {"temporal_mode": "invalid", "radius": 1},
        {"temporal_mode": "fused", "radius": 1},
        {"temporal_mode": "legacy", "radius": 1, "rolling_chunk": 4},
        {"temporal_mode": "legacy", "radius": 1, "rolling_cache_chunks": 2},
        {"temporal_mode": "rolling", "radius": 1, "rolling_chunk": 0},
        {"temporal_mode": "rolling", "radius": 1, "rolling_chunk": 65},
        {"temporal_mode": "rolling", "radius": 1, "rolling_cache_chunks": 0},
        {"temporal_mode": "rolling", "radius": 1, "rolling_cache_chunks": 65},
        {"temporal_mode": "rolling", "radius": 0},
    ],
)
def test_selector_rejects_invalid_combinations(arguments) -> None:
    source = patterned_clip(vs.GRAYS, 3)
    with pytest.raises(vs.Error):
        bm3dcuda.BM3Dv2(source, **arguments)


def test_default_is_explicit_rolling() -> None:
    source = patterned_clip(vs.GRAYS)
    arguments = {"radius": 2, "extractor_exp": 3, "fast": False}
    implicit = bm3dcuda.BM3Dv2(source, **arguments)
    explicit = bm3dcuda.BM3Dv2(source, temporal_mode="rolling", **arguments)
    assert_frames_equal(implicit, explicit, (0, 5, 10))


def test_default_radius_zero_uses_spatial_bm3d() -> None:
    source = patterned_clip(vs.GRAYS, 3)
    implicit = bm3dcuda.BM3Dv2(source, radius=0, fast=False, extractor_exp=3)
    explicit = bm3dcuda.BM3D(source, radius=0, fast=False, extractor_exp=3)
    assert_frames_equal(implicit, explicit, range(3))


def test_final_frame_of_int_max_length_clip() -> None:
    source = core.std.BlankClip(
        width=8,
        height=8,
        format=vs.GRAYS,
        length=np.iinfo(np.int32).max,
    )
    rolling = bm3dcuda.BM3Dv2(
        source,
        radius=1,
        temporal_mode="rolling",
        rolling_chunk=4,
        fast=False,
    )
    frame = rolling.get_frame(rolling.num_frames - 1)
    assert frame.width == 8


@pytest.mark.parametrize("radius", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [1, 3, 11])
def test_rolling_matches_deterministic_legacy(radius: int, length: int) -> None:
    source = patterned_clip(vs.GRAYS, length)
    arguments = {
        "radius": radius,
        "extractor_exp": 3,
        "block_step": 8,
        "bm_range": 4,
        "ps_range": 2,
    }
    legacy = bm3dcuda.BM3Dv2(
        source, temporal_mode="legacy", fast=False, **arguments
    )
    rolling = bm3dcuda.BM3Dv2(
        source,
        temporal_mode="rolling",
        rolling_chunk=4,
        fast=True,
        **arguments,
    )
    order = list(range(length)) + list(reversed(range(length)))
    assert_frames_equal(legacy, rolling, order)


def test_properties_and_unprocessed_planes_are_preserved() -> None:
    source = patterned_clip(vs.YUV444PS, 9)
    arguments = {
        "sigma": [3, 0, 0],
        "radius": 1,
        "extractor_exp": 3,
    }
    legacy = bm3dcuda.BM3Dv2(
        source, temporal_mode="legacy", **arguments
    )
    rolling = bm3dcuda.BM3Dv2(
        source, temporal_mode="rolling", rolling_chunk=4, **arguments
    )

    for n in (0, 3, 4, 8):
        source_frame = source.get_frame(n)
        legacy_frame = legacy.get_frame(n)
        rolling_frame = rolling.get_frame(n)
        assert rolling_frame.props["TestFrame"] == n
        assert np.array_equal(
            np.asarray(legacy_frame[0]), np.asarray(rolling_frame[0])
        )
        for plane in (1, 2):
            assert np.array_equal(
                np.asarray(source_frame[plane]),
                np.asarray(rolling_frame[plane]),
            )


def test_multi_chunk_cache_matches_legacy_on_random_access() -> None:
    source = patterned_clip(vs.GRAYS, 17)
    arguments = {
        "radius": 2,
        "extractor_exp": 3,
        "block_step": 8,
        "bm_range": 4,
        "ps_range": 2,
    }
    legacy = bm3dcuda.BM3Dv2(source, temporal_mode="legacy", **arguments)
    rolling = bm3dcuda.BM3Dv2(
        source,
        temporal_mode="rolling",
        rolling_chunk=4,
        rolling_cache_chunks=2,
        **arguments,
    )
    assert_frames_equal(legacy, rolling, (0, 4, 8, 4, 0, 12, 8, 16, 12, 4))


def test_concurrent_requests_match_serial_output() -> None:
    source = patterned_clip(vs.GRAYS, 17)
    arguments = {
        "radius": 3,
        "extractor_exp": 6,
        "block_step": 8,
        "bm_range": 4,
        "ps_range": 2,
        "temporal_mode": "rolling",
        "rolling_chunk": 4,
        "rolling_cache_chunks": 2,
    }
    serial = bm3dcuda.BM3Dv2(source, **arguments)
    concurrent = bm3dcuda.BM3Dv2(source, **arguments)
    order = (0, 4, 8, 12, 16, 1, 5, 9, 13)

    expected = {
        n: np.array(serial.get_frame(n)[0], copy=True)
        for n in order
    }
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            n: executor.submit(concurrent.get_frame, n)
            for n in order
        }
        actual = {
            n: np.array(future.result()[0], copy=True)
            for n, future in futures.items()
        }

    for n in order:
        assert np.array_equal(expected[n], actual[n]), n
