"""Independent, frame-aware confirmation tracks for the fast D435 scanner."""

from dataclasses import dataclass


def rect_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, aw * ah + bw * bh - intersection)


@dataclass
class BarcodeTrack:
    value: str
    rect: tuple[int, int, int, int]
    first_rect: tuple[int, int, int, int]
    hits: int
    last_at: float
    frame_id: int


class BarcodeEvidenceTracks:
    """YOLO views share a track; scanner, unverified and stripe tracks do not."""

    def __init__(self):
        self.tracks: dict[str, BarcodeTrack] = {}

    def clear(self):
        self.tracks.clear()

    def observe(self, hit, *, now, frame_id, max_gap, min_iou, match=None):
        family = "yolo" if hit.source.startswith("yolo") else hit.source
        previous = self.tracks.get(family)
        if previous is not None and frame_id <= previous.frame_id:
            return previous
        same = (
            previous is not None and previous.value == hit.value
            and 0.0 <= now - previous.last_at <= max_gap
            and (match(previous.rect, hit.rect) if match else
                 rect_iou(previous.rect, hit.rect) >= min_iou)
        )
        track = BarcodeTrack(
            value=hit.value, rect=tuple(hit.rect),
            first_rect=previous.first_rect if same else tuple(hit.rect),
            hits=previous.hits + 1 if same else 1,
            last_at=now, frame_id=frame_id,
        )
        self.tracks[family] = track
        return track
