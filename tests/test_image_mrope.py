import numpy as np
import pytest

from jevision_decide.image import smart_resize
from jevision_decide.mrope import position_ids


def test_small_image_is_scaled_up_to_the_minimum_pixel_count():
    # a 224 px thumbnail becomes 256 px (65536 px minimum), i.e. 64 tokens after the 2x2 merge
    h, w = smart_resize(224, 224, 32, 65536, 16777216)
    assert (h, w) == (256, 256)


def test_extreme_aspect_ratio_is_rejected():
    with pytest.raises(ValueError):
        smart_resize(10, 3000, 32, 65536, 16777216)


def test_text_positions_count_up_on_all_three_axes():
    pos = position_ids([1, 2, 3, 4], None, image_token_id=99)
    assert pos.shape == (3, 1, 4)
    assert (pos[:, 0, :] == np.arange(4)).all()
