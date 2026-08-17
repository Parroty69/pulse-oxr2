from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class TileRecord:
    tile_index: int
    image: np.ndarray
    full_image_bbox: list[int]


class TileCoordinateMapper:
    def __init__(self, image_shape: tuple[int, int], tile_size: int, overlap: float):
        self.height, self.width = image_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = max(1, int(round(tile_size * (1.0 - overlap))))
        self._bboxes = _generate_tile_bboxes(self.width, self.height, tile_size, self.stride)

    @property
    def bboxes(self) -> list[list[int]]:
        return [bbox.copy() for bbox in self._bboxes]

    def local_to_full(self, tile_index: int, local_xy: tuple[int, int]) -> tuple[int, int]:
        x0, y0, _, _ = self._bboxes[tile_index]
        lx, ly = local_xy
        return x0 + lx, y0 + ly

    def full_to_local(self, tile_index: int, full_xy: tuple[int, int]) -> tuple[int, int]:
        x0, y0, _, _ = self._bboxes[tile_index]
        fx, fy = full_xy
        return fx - x0, fy - y0


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _generate_tile_bboxes(width: int, height: int, tile_size: int, stride: int) -> list[list[int]]:
    x_starts = _axis_starts(width, tile_size, stride)
    y_starts = _axis_starts(height, tile_size, stride)
    bboxes: list[list[int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(width, x0 + tile_size)
            y1 = min(height, y0 + tile_size)
            bboxes.append([x0, y0, x1, y1])
    return bboxes


def build_dual_stream(image_np: np.ndarray, tile_size: int = 1024, overlap: float = 0.2, global_size: int = 512) -> tuple[np.ndarray, list[TileRecord], TileCoordinateMapper]:
    mapper = TileCoordinateMapper(image_np.shape[:2], tile_size=tile_size, overlap=overlap)

    tiles: list[TileRecord] = []
    for idx, (x0, y0, x1, y1) in enumerate(mapper.bboxes):
        tile = image_np[y0:y1, x0:x1]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            padded = np.zeros((tile_size, tile_size), dtype=image_np.dtype)
            padded[: tile.shape[0], : tile.shape[1]] = tile
            tile = padded
        tiles.append(TileRecord(tile_index=idx, image=tile, full_image_bbox=[x0, y0, x1, y1]))

    global_thumbnail = np.asarray(
        Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8)).resize((global_size, global_size))
    )
    return global_thumbnail, tiles, mapper
