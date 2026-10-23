"""No camera/GPU/ROS initialization: synchronized sets and capture-clock checks."""

import ast
from collections import deque
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dobot_nova5_driver.d405_pipeline_performance import (
    latest_synchronized_frames, LocalizationTiming,
)


def frames(number):
    return SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_frame_number=lambda: number),
        infrared_left=('left', number), infrared_right=('right', number), depth=('depth', number),
    )


def test_latest_capture_keeps_one_complete_synchronized_set():
    first, second, newest = frames(1), frames(2), frames(3)
    queue = deque([second, newest, None])
    pipeline = SimpleNamespace(wait_for_frames=lambda: first, poll_for_frames=queue.popleft)
    assert latest_synchronized_frames(pipeline) is newest
    assert newest.infrared_left == ('left', 3)
    assert newest.infrared_right == ('right', 3)
    assert newest.depth == ('depth', 3)


def test_duplicate_and_regressing_color_frames_do_not_replace_newer_capture():
    first, newer = frames(3), frames(4)
    queue = deque([newer, frames(4), frames(2), None])
    pipeline = SimpleNamespace(wait_for_frames=lambda: first, poll_for_frames=queue.popleft)
    assert latest_synchronized_frames(pipeline) is newer


def test_continuously_ready_stream_cannot_make_capture_drain_unbounded():
    count = [1]

    def poll():
        count[0] += 1
        return frames(count[0])

    pipeline = SimpleNamespace(wait_for_frames=lambda: frames(1), poll_for_frames=poll)
    assert latest_synchronized_frames(pipeline).get_color_frame().get_frame_number() == 33
    assert count == [33]


def test_old_sdk_without_poll_keeps_existing_wait_behavior():
    first = frames(3)
    assert latest_synchronized_frames(SimpleNamespace(wait_for_frames=lambda: first)) is first


@pytest.fixture(params=[4, 5])
def capture_clock(request):
    path = Path(__file__).resolve().parents[2] / 'Fast-FoundationStereoPose-dul_cam/dul_cam' / (
        f'd405_cosmetic_box_leftmost_height75_panel_v{request.param}.py')
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'frame_capture_stamp')
    namespace = dict(np=np, logging=logging, camera_clock_offset_samples=deque(maxlen=30),
                     camera_timestamp_source='host_return', camera_timestamp_logged=False,
                     rs=SimpleNamespace(timestamp_domain=SimpleNamespace(system_time='system', global_time='global')))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)

    def convert(capture_s, returned_s):
        color = SimpleNamespace(get_timestamp=lambda: capture_s*1000,
                                get_frame_timestamp_domain=lambda: 'system')
        return namespace['frame_capture_stamp'](SimpleNamespace(get_color_frame=lambda: color), returned_s)

    return convert


def test_old_capture_time_is_preserved_instead_of_relabelled_as_arrival(capture_clock):
    stamp, source = capture_clock(1000., 1000.985)
    assert stamp == 1000.
    assert source == 'd405_system_time'


@pytest.mark.parametrize('capture_s', [0., -1., float('nan'), float('inf'), 1001.])
def test_invalid_or_future_capture_time_cannot_look_fresh(capture_clock, capture_s):
    stamp, source = capture_clock(capture_s, 1000.)
    assert stamp == 0.
    assert source == 'invalid_capture_time'


def test_invalid_capture_warning_is_limited_until_clock_recovers(capture_clock, caplog):
    with caplog.at_level(logging.WARNING):
        capture_clock(0., 1000.)
        capture_clock(0., 1000.)
        capture_clock(999.9, 1000.)
        capture_clock(0., 1000.)
    assert sum('pose publication blocked' in record.message for record in caplog.records) == 2


def test_frame_timing_continues_after_stable_target_and_reports_stage_latency():
    now = [1.]
    timing = LocalizationTiming(clock=lambda: now[0])
    timing.begin_frame()
    now[0] = 1.1
    timing.record('sam_track', 1.)
    now[0] = 1.8
    timing.record('stereo_cloud', 1.1)
    report = timing.slow_frame_report(capture_s=1000., returned_s=1000.02, now_epoch=1000.985)
    assert report == dict(capture_age_ms=985., age_at_camera_read_ms=20., loop_ms=800.,
                          stage_ms={'sam_track': 100., 'stereo_cloud': 700.})
    assert not timing.active
    assert timing.slow_frame_report(capture_s=1000., returned_s=1000.02, now_epoch=1001.) is None


def test_fast_frame_has_no_slow_log_and_repeated_geometry_is_not_double_counted():
    now = [1.]
    timing = LocalizationTiming(clock=lambda: now[0])
    timing.begin_frame()
    now[0] = 1.1
    timing.record('geometry', 1.)
    now[0] = 1.2
    timing.record('geometry', 1.)
    assert timing.frame_stages['geometry'] == pytest.approx(.2)
    assert timing.slow_frame_report(capture_s=1000., returned_s=1000.02, now_epoch=1000.2) is None
