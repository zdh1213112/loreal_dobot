#!/usr/bin/env python3
"""Manually test one controller DO on the 101 Nova5 Dashboard (port 29999).

Run this only while the normal 101 control node is stopped, and with the
turntable area clear. This script tests an output level, not motor safety.
"""

import argparse
import re
import socket
import sys


DEFAULT_ROBOT_IP = "192.168.111.101"
DASHBOARD_PORT = 29999
REPLY_PATTERN = re.compile(r"^\s*(-?\d+)\s*,\s*\{([^}]*)\}")


def parse_reply(reply: str) -> tuple[int, list[str]]:
    match = REPLY_PATTERN.match(reply)
    if not match:
        raise RuntimeError(f"无法解析控制器回复: {reply!r}")
    error_id = int(match.group(1))
    values = [item.strip() for item in match.group(2).split(",") if item.strip()]
    if error_id != 0:
        raise RuntimeError(f"控制器返回错误码 {error_id}: {reply.strip()}")
    return error_id, values


def exchange(connection: socket.socket, command: str) -> list[str]:
    connection.sendall(command.encode("utf-8"))
    reply = connection.recv(1024).decode("utf-8", errors="replace")
    if not reply:
        raise RuntimeError(f"控制器未回复 {command}")
    print(f"发送: {command}  回复: {reply.strip()}")
    _, values = parse_reply(reply)
    return values


def read_do(connection: socket.socket, index: int) -> int:
    values = exchange(connection, f"GetDO({index})")
    if len(values) != 1 or values[0] not in ("0", "1"):
        raise RuntimeError(f"GetDO({index}) 回复没有有效的 0/1 状态: {values!r}")
    return int(values[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 101 控制器 DO 开/关/读取")
    parser.add_argument("--do", type=int, required=True, choices=range(1, 9),
                        metavar="1..8", help="控制器 DO 编号，现场核对接线后填写")
    parser.add_argument("--state", choices=("read", "on", "off"), default="read",
                        help="read 只读取（默认），on 输出 1，off 输出 0")
    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP, help="101 控制器 IP")
    parser.add_argument("--timeout", type=float, default=3.0, help="TCP 超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不连接机械臂")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    command = (f"DOInstant({args.do},{1 if args.state == 'on' else 0})"
               if args.state != "read" else f"GetDO({args.do})")
    if args.dry_run:
        print(f"仅预览: {args.ip}:{DASHBOARD_PORT}  {command}")
        if args.state != "read":
            print(f"随后读取: GetDO({args.do})")
        return 0

    if args.state != "read":
        print("注意：这会改变实际 DO 电平。请先停止 101 正常控制节点，并确认转盘区域安全。")
    try:
        with socket.create_connection((args.ip, DASHBOARD_PORT), timeout=args.timeout) as connection:
            connection.settimeout(args.timeout)
            if args.state != "read":
                exchange(connection, command)
            actual = read_do(connection, args.do)
    except (OSError, RuntimeError) as exc:
        print(f"测试失败: {exc}", file=sys.stderr)
        return 1

    print(f"DO {args.do} 当前状态: {'ON (1)' if actual else 'OFF (0)'}")
    if args.state != "read" and actual != (1 if args.state == "on" else 0):
        print("警告：读回状态与请求不一致；不要据此判断转盘已经启动或停止。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
