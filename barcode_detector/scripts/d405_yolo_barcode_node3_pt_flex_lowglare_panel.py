#!/usr/bin/env python3
from d405_yolo_pt_common_lowglare_panel import D405YoloPtBase, spin_node


class BarcodeDetectorNodePtFlex(D405YoloPtBase):
    window_name = "D405 RGB + SAM2"
    enable_shape_filter = False
    enable_mouse_roi = True

    def __init__(self):
        super().__init__(
            node_name="barcode_detector_node_pt_flex",
            startup_label="D405 启动：640x480 + 低反光手动曝光（YOLO .pt Flex 检测）",
        )


def main():
    spin_node(BarcodeDetectorNodePtFlex)


if __name__ == "__main__":
    main()
