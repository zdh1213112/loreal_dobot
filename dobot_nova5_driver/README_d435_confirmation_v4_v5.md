# V4 / V5 D435 侧面条码确认

本轮优化保留 launch 中的 `turntable_d435_confidence:=0.40`、完整推理视野、局部/分块回退，以及 YOLO 和 OpenCV 两条独立检测路径。

## 确认与疑似候选

- 每张图同时检查 YOLO 和 OpenCV。YOLO 的 detail/full/tile 结果属于同一路；OpenCV、未通过结构校验的 OpenCV 候选和运动条纹各自计数。不同来源不能互相凑够确认次数，同一帧也不能重复计数。
- YOLO 仍默认要求 2 次同位置有效命中，最大间隔 0.70 秒、IoU 至少 0.15。原有置信度、ROI、面积、细节和平行条纹条件保持。
- OpenCV 独立确认仍默认要求 3 次有效命中，最大间隔 0.25 秒、IoU 至少 0.20；新增细节/平行条纹检查，倾斜候选先使用检测四边形做透视校正后再次检查，不强制要求 YOLO 同时检测到。
- OpenCV 框如果结构不清晰，保留为 `scanner_unverified` 疑似候选，不能发布 `success` 或连续存在确认。短暂漏一帧不会立即丢弃候选，它按时间自动过期。
- 同位置匹配失败、扫描重新启用或 ROI 改变会隔离旧证据。旧扫描推理回包不能借连续检测开关进入新扫描。

## 转盘补拍

候选通过独立的 `/turntable_barcode_candidate` 话题报告。该话题只请求更清晰的观察，不能授权抓取；侧码确认仍由原来的 `/turntable_barcode_result` 发布。

正常候选先按原有帧数快速确认。候选持续未确认时，V4/V5 使用现有可靠的停盘脉冲暂停转盘，并额外观察少量静止帧。收到侧码或顶部确认就保持停止；仍未确认则在检查 102 净空后继续剩余旋转搜索。

候选停盘/观察/重启耗时不占原旋转搜索预算，也不提前触发 D405 抓取。最终停止后，继续重新采集 D405 的抓取目标。取消、换料或关闭扫描会中止补拍，不会自动重启转盘。

| launch 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `turntable_d435_confidence` | `0.40` | YOLO 候选置信度 |
| `turntable_d435_yolo_stable_hits` | `2` | YOLO 独立确认次数 |
| `turntable_d435_scanner_assist_hits` | `3` | OpenCV 独立确认次数 |
| `turntable_candidate_grace_s` | `0.12` | 持续疑似候选进入补拍前，留给正常确认的时间 |
| `turntable_candidate_recheck_s` | `0.35` | 停盘脉冲完成后的额外观察上限；范围 0–2 秒 |
| `turntable_candidate_max_rechecks` | `2` | 一次物料扫描最多补拍次数；设为 0 可关闭候选停盘 |

额外观察时间不包括现有停止/启动脉冲。仅在疑似候选持续出现时产生补拍等待，已有成功确认立即进入原有停盘流程。

例如保留检测优化、关闭额外停盘：

```bash
ros2 launch dobot_nova5_driver cosmetic_box_single_arm_cycle_v5_d435.launch.py \
  turntable_candidate_max_rechecks:=0
```

V4 使用对应的 `cosmetic_box_single_arm_cycle_v4_d435.launch.py`。

## 验证范围

离线测试覆盖两路独立确认、混合来源不累计、同帧去重、暗光/轻度模糊/旋转条纹、无条纹候选保留、过期和跨轮数据拒收，以及 V4/V5 补拍成功、超时恢复、取消和完整旋转预算。

对现有样本目录均匀抽取 32 张原图进行 CPU 回放，7 张出现 YOLO 候选，优化后均保留；该组图片没有触发旧 OpenCV 候选，也没有人工真假标签，因此不构成实际召回率或误检率结论。现场仍应比较模糊、反光真条码和已知误检场景，并观察新增双路推理对帧率的影响。
