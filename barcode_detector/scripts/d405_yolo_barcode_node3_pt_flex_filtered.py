#!/usr/bin/env python3
from d405_yolo_pt_common import D405YoloPtBase, spin_node


class BarcodeDetectorNodePtFlexFiltered(D405YoloPtBase):
    window_name = "YOLO PT Flex Detection Filtered"
    enable_shape_filter = True

    def __init__(self):
        super().__init__(
            node_name="barcode_detector_node_pt_flex_filtered",
            startup_label="D405 启动：1280x720 + 1x2 切片（YOLO .pt Flex + 条码形状过滤）",
        )


def main():
    spin_node(BarcodeDetectorNodePtFlexFiltered)


if __name__ == "__main__":
    main()
