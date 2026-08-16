import numpy as np

from src.ingestion.tiling import TileCoordinateMapper, build_dual_stream


def test_tile_coordinate_round_trip_and_overlap():
    image = np.zeros((3000, 3000), dtype=np.float32)
    _, tiles, mapper = build_dual_stream(image, tile_size=1024, overlap=0.2)

    assert len(tiles) > 1

    x0, y0, _, _ = mapper.bboxes[0]
    full = mapper.local_to_full(0, (50, 75))
    local = mapper.full_to_local(0, full)
    assert local == (50, 75)
    assert full == (x0 + 50, y0 + 75)

    first = mapper.bboxes[0]
    second = mapper.bboxes[1]
    overlap_px = first[2] - second[0]
    expected_overlap = 1024 - mapper.stride
    assert overlap_px == expected_overlap
