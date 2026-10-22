"""Socket cleanup regressions for the active 101/102 SDK copies; no hardware IO."""

import ast
import contextlib
import errno
import importlib.util
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SDK_PATHS = {
    '102': os.environ.get('DOBOT_SDK_102_SOURCE'),
    '101': Path(os.environ.get('DOBOT_SDK_101_SOURCE',
        str(PACKAGE_ROOT / 'dobot_nova5_driver/TCP_IP_Python_V4/dobot_api.py'))),
}


@pytest.fixture(scope='module', params=['102', '101'])
def sdk(request):
    source = SDK_PATHS[request.param]
    if source is None:
        try:
            installed = importlib.util.find_spec('lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api')
        except ModuleNotFoundError:
            installed = None
        if installed is None:
            pytest.skip('Optional 102 LeRobot SDK is not installed in this test environment')
        source = installed.origin
    spec = importlib.util.spec_from_file_location('cleanup_sdk_' + request.param, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSocket:
    def __init__(self, *, shutdown_errno=None, close_errno=None):
        self.shutdown_errno, self.close_errno = shutdown_errno, close_errno
        self.shutdown_calls = self.close_calls = 0
        self.closed = False

    def connect(self, address):
        self.address = address  # Deliberately no real socket/robot connection.

    def setsockopt(self, *_args):
        pass

    def shutdown(self, _how):
        self.shutdown_calls += 1
        if self.shutdown_errno is not None:
            raise OSError(self.shutdown_errno, os.strerror(self.shutdown_errno))
        if self.closed:
            raise OSError(errno.EBADF, os.strerror(errno.EBADF))

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.close_errno is not None:
            raise OSError(self.close_errno, os.strerror(self.close_errno))


@pytest.fixture
def api_factory(sdk, monkeypatch):
    created = []

    def create(channel=None):
        channel = FakeSocket() if channel is None else channel
        monkeypatch.setattr(sdk.socket, 'socket', lambda: channel)
        api = sdk.DobotApiDashboard('192.0.2.102', 29999)
        created.append(api)
        return api, channel

    yield create
    # Keep objects alive until here, then disarm SDK __del__ after fake cleanup.
    for api in created:
        api.socket_dobot = 0


def test_explicit_close_repeated_close_and_destructor_release_the_socket_once(api_factory, capsys):
    api, channel = api_factory()
    api.close()
    api.close()
    api.__del__()
    assert channel.shutdown_calls == channel.close_calls == 1
    assert api.socket_dobot == 0
    assert not capsys.readouterr().out


@pytest.mark.parametrize('error', [errno.ENOTCONN, errno.EBADF])
def test_expected_shutdown_error_still_closes_and_is_quiet(api_factory, capsys, error):
    api, channel = api_factory(FakeSocket(shutdown_errno=error))
    api.close()
    api.__del__()
    assert channel.close_calls == 1 and channel.closed
    assert api.socket_dobot == 0
    assert not capsys.readouterr().out


def test_socket_closed_elsewhere_is_not_reported_as_a_new_connection_fault(api_factory, capsys):
    api, channel = api_factory()
    channel.close()
    api.close()
    api.__del__()
    assert api.socket_dobot == 0
    assert channel.shutdown_calls == 1 and channel.close_calls == 2
    assert not capsys.readouterr().out


def test_unexpected_shutdown_error_remains_visible_and_does_not_leak(api_factory, capsys):
    api, channel = api_factory(FakeSocket(shutdown_errno=errno.EIO))
    api.close()
    api.__del__()
    output = capsys.readouterr().out
    assert 'Error while closing socket' in output and 'Input/output error' in output
    assert channel.close_calls == 1 and api.socket_dobot == 0


@pytest.mark.parametrize('error,visible', [(errno.EBADF, False), (errno.EIO, True)])
def test_close_error_is_only_suppressed_for_an_already_invalid_descriptor(api_factory, capsys, error, visible):
    api, channel = api_factory(FakeSocket(close_errno=error))
    api.close()
    api.__del__()
    assert bool(capsys.readouterr().out) is visible
    assert channel.shutdown_calls == channel.close_calls == 1
    assert api.socket_dobot == 0


def test_destructor_is_safe_when_construction_has_not_created_any_socket(sdk, capsys):
    api = sdk.DobotApiDashboard.__new__(sdk.DobotApiDashboard)
    try:
        api.close()
        api.__del__()
        assert api.socket_dobot == 0
        assert not capsys.readouterr().out
    finally:
        api.socket_dobot = 0


def test_simultaneous_close_does_not_wait_on_the_command_lock_or_close_twice(api_factory, capsys):
    api, channel = api_factory()
    entered, release, second_done = threading.Event(), threading.Event(), threading.Event()
    shutdown = channel.shutdown
    failures = []

    def slow_shutdown(how):
        entered.set()
        assert release.wait(2.)
        shutdown(how)

    channel.shutdown = slow_shutdown

    def close(done=None):
        try:
            api.close()
        except BaseException as error:
            failures.append(error)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=close)
    second = threading.Thread(target=close, args=(second_done,))
    try:
        # A stalled command must not prevent detaching/closing a dead connection.
        with api._DobotApi__globalLock:
            first.start()
            assert entered.wait(1.)
            second.start()
            assert second_done.wait(1.)
    finally:
        release.set()
        first.join(2.)
        if second.ident is not None:
            second.join(2.)
    api.__del__()
    assert not first.is_alive() and not second.is_alive() and not failures
    assert channel.shutdown_calls == channel.close_calls == 1
    assert not capsys.readouterr().out


def test_reentrant_cleanup_returns_without_deadlocking_or_reclosing(api_factory, capsys):
    api, channel = api_factory()
    shutdown = channel.shutdown

    def reentrant(how):
        api.__del__()
        shutdown(how)

    channel.shutdown = reentrant
    api.close()
    assert api.socket_dobot == 0
    assert channel.shutdown_calls == channel.close_calls == 1
    assert not capsys.readouterr().out


def test_dedicated_dashboard_failure_keeps_shared_fallback_and_primary_socket(api_factory, capsys):
    # Execute the real gripper fallback method, without importing/starting a robot.
    try:
        installed = importlib.util.find_spec('lerobot.robots.dobot_nova5_dh.dobot_nova5_dh')
    except ModuleNotFoundError:
        installed = None
    if installed is None:
        pytest.skip('Optional 102 gripper fallback adapter is not installed')
    source = Path(installed.origin)
    tree = ast.parse(source.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'DobotNova5DH')
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == '_create_gripper_modbus')
    primary, primary_socket = api_factory()
    dedicated, dedicated_socket = api_factory()
    shared_adapter = object()
    calls = []

    def modbus(api, *_args):
        calls.append(api)
        if api is dedicated:
            raise RuntimeError("Unexpected Dobot response: 'Connection refused, IP:Port has been occupied'")
        assert api is primary
        return shared_adapter

    namespace = dict(contextlib=contextlib, DeviceNotConnectedError=RuntimeError,
                     DobotApiDashboard=lambda *_args: dedicated, _DobotModbusRTU=modbus)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    logger = SimpleNamespace(info=Mock(), warn=Mock())
    robot = SimpleNamespace(_robot=primary, _gripper_robot=None, logger=logger,
        config=SimpleNamespace(dedicated_gripper_dashboard=True, robot_ip='192.0.2.102', dashboardPort=29999,
                               master_ip='127.0.0.1', master_port=60000, dh_gripper=SimpleNamespace(slave_id=1)))
    assert namespace['_create_gripper_modbus'](robot) is shared_adapter
    dedicated.__del__()  # The rejected channel can later be garbage-collected safely.
    assert calls == [dedicated, primary]
    assert dedicated_socket.shutdown_calls == dedicated_socket.close_calls == 1
    assert robot._gripper_robot is None
    assert primary.socket_dobot is primary_socket and not primary_socket.closed
    logger.warn.assert_called_once()
    assert 'Connection refused' in logger.warn.call_args.args[0]
    assert not capsys.readouterr().out
