#!/usr/bin/env python3
"""101 控制器 DO 手动测试界面；运行：python3 scripts/test_turntable_do_gui.py。

只用于接线调试。启动界面不会自动输出；关闭界面不会自动关闭 DO。
正常的 101 控制节点应先停止，转盘区域应保持无人、无机械臂进入。
"""

import socket
import sys
import time

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
        QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
        QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
    )

from test_turntable_do import (
    DASHBOARD_PORT, DEFAULT_ROBOT_IP, exchange, read_do,
)


class DoRequest(QThread):
    result = Signal(object)

    def __init__(self, index: int, action: str, pulse_ms: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.action = action
        self.pulse_ms = pulse_ms

    def run(self):
        output_may_be_on = False
        try:
            with socket.create_connection(
                (DEFAULT_ROBOT_IP, DASHBOARD_PORT), timeout=3.0
            ) as connection:
                connection.settimeout(3.0)
                if self.action == "pulse":
                    delay_s = self.pulse_ms / 1000.0
                    exchange(connection, f"DOInstant({self.index},0)")
                    time.sleep(delay_s)
                    output_may_be_on = True
                    exchange(connection, f"DOInstant({self.index},1)")
                    time.sleep(delay_s)
                    exchange(connection, f"DOInstant({self.index},0)")
                    output_may_be_on = False
                elif self.action != "read":
                    value = 1 if self.action == "on" else 0
                    output_may_be_on = self.action == "on"
                    exchange(connection, f"DOInstant({self.index},{value})")
                actual = read_do(connection, self.index)
            self.result.emit((self.index, self.action, actual, output_may_be_on, ""))
        except (OSError, RuntimeError) as exc:
            self.result.emit((self.index, self.action, None, output_may_be_on, str(exc)))


class DoTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.active_index = None  # 读回 ON，或输出 1 后状态不明的 DO
        self.setWindowTitle("101 控制器 DO 手动测试")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("连接目标：101 机械臂控制器 192.168.111.101:29999"))

        group = QGroupBox("选择控制器输出口")
        form = QFormLayout(group)
        self.index_input = QSpinBox()
        self.index_input.setRange(1, 8)
        self.index_input.setValue(1)
        form.addRow("DO 编号（核对接线）：", self.index_input)
        self.pulse_input = QSpinBox()
        self.pulse_input.setRange(50, 5000)
        self.pulse_input.setValue(300)
        self.pulse_input.setSuffix(" ms")
        form.addRow("脉冲复位/保持时间：", self.pulse_input)
        layout.addWidget(group)

        self.pulse_button = QPushButton("脉冲触发（每次切换启/停）")
        self.pulse_button.clicked.connect(lambda: self.request("pulse"))
        layout.addWidget(self.pulse_button)

        row = QHBoxLayout()
        self.read_button = QPushButton("读取状态")
        self.on_button = QPushButton("输出 1")
        self.off_button = QPushButton("输出 0 / 停止请求")
        self.read_button.clicked.connect(lambda: self.request("read"))
        self.on_button.clicked.connect(lambda: self.request("on"))
        self.off_button.clicked.connect(lambda: self.request("off"))
        row.addWidget(self.read_button)
        row.addWidget(self.on_button)
        row.addWidget(self.off_button)
        layout.addLayout(row)

        self.status = QLabel("未读取；界面启动不会自动改变 DO。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        warning = QLabel(
            "注意：输出 1/0 是 DO 逻辑电平，不保证转盘实际启停；"
            "关闭界面不会自动输出 0。测试前停止正常的 101 控制节点并清空转盘区域。"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def set_busy(self, busy: bool):
        self.read_button.setEnabled(not busy)
        self.pulse_button.setEnabled(not busy)
        self.on_button.setEnabled(not busy)
        self.off_button.setEnabled(not busy)
        self.index_input.setEnabled(not busy and self.active_index is None)
        self.pulse_input.setEnabled(not busy)

    def request(self, action: str):
        if self.worker is not None:
            return
        index = self.index_input.value()
        if action in ("on", "pulse"):
            message = (
                f"将对 DO{index} 自动发送 0→1→0 脉冲。\n"
                "每点击一次，转盘控制器会在启动/停止之间切换。"
                if action == "pulse" else
                f"将 101 控制器 DO{index} 设置为 1。"
            )
            answer = QMessageBox.question(
                self,
                "确认触发转盘" if action == "pulse" else "确认输出 1",
                message + "\n转盘可能立即运动；确认区域安全后继续。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.status.setText(f"正在连接 101 并处理 DO{index}：{action} …")
        self.set_busy(True)
        self.worker = DoRequest(index, action, self.pulse_input.value(), self)
        self.worker.result.connect(self.show_result)
        self.worker.finished.connect(self.finish_request)
        self.worker.start()

    def show_result(self, result):
        index, action, actual, output_may_be_on, error = result
        if error:
            if output_may_be_on:
                self.active_index = index
            self.status.setText(
                f"DO{index} 操作失败：{error}。"
                + ("输出 1 可能已生效，状态未知；请尝试输出 0 并现场确认。"
                   if output_may_be_on else "")
            )
            return
        if actual == 1 or (action == "on" and actual != 1):
            self.active_index = index
        elif self.active_index == index:
            self.active_index = None
        expected = 1 if action == "on" else 0
        mismatch = action != "read" and actual != expected
        if action == "pulse" and not mismatch:
            self.status.setText(
                f"DO{index} 的 0→1→0 触发脉冲已发送，输出已复位为 0。"
                "转盘实际处于运行还是停止，需要现场观察或反馈信号确认。"
            )
        else:
            self.status.setText(
                f"DO{index} 读回状态：{'1 / ON' if actual else '0 / OFF'}。"
                + ("与请求不一致，请检查控制器回复和接线。" if mismatch else "")
                + "读回状态不代表转盘已停稳。"
            )

    def finish_request(self):
        worker = self.worker
        self.worker = None
        self.set_busy(False)
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event):
        if self.worker is not None:
            QMessageBox.information(self, "请稍候", "通信尚未结束，请等结果返回后再关闭。")
            event.ignore()
            return
        if self.active_index is not None:
            answer = QMessageBox.question(
                self,
                "DO 可能仍为 1",
                f"DO{self.active_index} 已读回 1 或状态未知。\n"
                "关闭界面不会自动输出 0，转盘可能继续运动。仍要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = DoTestWindow()
    window.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
