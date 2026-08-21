#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULE_SOURCE="${SCRIPT_DIR}/../config/70-cosmetic-barcode-scanner.rules"
RULE_TARGET="/etc/udev/rules.d/70-cosmetic-barcode-scanner.rules"

if [[ ! -f "${RULE_SOURCE}" ]]; then
  echo "找不到规则文件: ${RULE_SOURCE}" >&2
  exit 1
fi

sudo install -m 0644 "${RULE_SOURCE}" "${RULE_TARGET}"
sudo udevadm control --reload-rules

echo "扫码枪持久权限规则已安装: ${RULE_TARGET}"
echo "请拔插一次扫码枪；ROS 节点会在 2 秒内自动重新连接。"
