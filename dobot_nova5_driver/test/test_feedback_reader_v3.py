"""V3 feedback framing tests; no robot connection is opened."""

import numpy as np
import pytest

from dobot_nova5_driver.TCP_IP_Python_V4.dobot_api import MyType
import dobot_nova5_driver.controller_v3 as controller_v3
from dobot_nova5_driver.controller_v3 import (
    DobotNova5Controller,
    FramedDobotApiFeedBack,
)


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def recv(self, _size):
        return next(self.chunks)

    def shutdown(self, _how):
        pass

    def close(self):
        pass


def make_frame(timestamp):
    packet = np.zeros(1, dtype=MyType)
    packet["len"] = MyType.itemsize
    packet["TimeStamp"] = timestamp
    packet["RobotMode"] = 5
    return packet.tobytes()


def make_reader(chunks):
    reader = FramedDobotApiFeedBack.__new__(FramedDobotApiFeedBack)
    reader.socket_dobot = FakeSocket(chunks)
    reader._feedback_buffer = bytearray()
    return reader


def test_split_tcp_reads_reconstruct_one_feedback_frame():
    frame = make_frame(123456)
    reader = make_reader([frame[:1], frame[1:30], frame[30:700], frame[700:]])

    assert int(reader.feedBackData()["TimeStamp"][0]) == 123456
    assert reader._feedback_buffer == bytearray()


def test_coalesced_frames_use_latest_and_preserve_partial_next_frame():
    first = make_frame(100)
    second = make_frame(200)
    third = make_frame(300)
    reader = make_reader([first + second + third[:517], third[517:]])

    assert int(reader.feedBackData()["TimeStamp"][0]) == 200
    assert len(reader._feedback_buffer) == 517
    assert int(reader.feedBackData()["TimeStamp"][0]) == 300


def test_reader_resynchronizes_after_truncated_frame():
    frame = make_frame(900)
    reader = make_reader([b"\x00" * 80 + frame[:1], frame[1:]])

    assert int(reader.feedBackData()["TimeStamp"][0]) == 900


def test_closed_feedback_socket_is_reported():
    reader = make_reader([b""])

    with pytest.raises(ConnectionError, match="feedback socket closed"):
        reader.feedBackData()


def test_transient_feedback_error_retries_without_using_full_safety_budget(monkeypatch):
    controller = DobotNova5Controller("192.0.2.102")
    packet = np.frombuffer(make_frame(700), dtype=MyType)

    class TransientFeedback:
        calls = 0

        def feedBackData(self):
            self.calls += 1
            if self.calls == 1:
                raise OSError("transient read")
            controller._stop_feedback.set()
            return packet

    controller.feedback = TransientFeedback()
    sleeps = []
    monkeypatch.setattr(controller_v3.time, "sleep", sleeps.append)

    controller._feedback_loop()

    assert sleeps == [controller_v3.FEEDBACK_READ_ERROR_RETRY_S]
    assert controller.feedback_data is packet
    assert controller.feedback_last_read_error == ""
