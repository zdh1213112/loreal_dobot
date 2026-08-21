#!/usr/bin/env bash
set -euo pipefail

SCANNER_LINK="${SCANNER_DEVICE_LINK:-/dev/input/by-id/usb-Linux_3.0.8_with_fh_otg_BTW_Hid_Device-event-kbd}"

if [[ ! -e "${SCANNER_LINK}" ]]; then
  echo "未找到扫码枪输入设备: ${SCANNER_LINK}" >&2
  echo "请先插入扫码枪；扫码节点会在设备恢复后自动重连。" >&2
  exit 1
fi

SCANNER_EVENT="$(readlink -f "${SCANNER_LINK}")"
TARGET_USER="${SUDO_USER:-${USER}}"

if ! command -v setfacl >/dev/null 2>&1; then
  echo "缺少 setfacl，请先安装 acl 软件包: sudo apt install acl" >&2
  exit 1
fi

echo "扫码枪输入设备: ${SCANNER_EVENT}"
if [[ "${EUID}" -eq 0 ]]; then
  setfacl -m "u:${TARGET_USER}:r" "${SCANNER_EVENT}"
else
  sudo setfacl -m "u:${TARGET_USER}:r" "${SCANNER_EVENT}"
fi

echo "已授权用户 ${TARGET_USER} 读取扫码枪输入设备。"
echo "正在运行的 ROS 扫码节点会在 2 秒内自动连接，无需重启 launch。"
echo "如需拔插后仍自动授权，请执行项目中的 install_cosmetic_barcode_udev_rule.sh。"
