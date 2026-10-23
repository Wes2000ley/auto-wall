import numpy as np
import pytest

from src.ml.light_data import cutmix_light_crops


def labelled_crops():
    targets = np.random.default_rng(11).random((4, 96, 96), dtype=np.float32)
    images = np.repeat(np.repeat(targets, 4, axis=1), 4, axis=2)[:, None].repeat(3, axis=1)
    # A separate channel identifies the donor of each pixel.
    images[:, 1] = np.arange(4)[:, None, None] / 4
    valid = np.ones_like(targets, dtype=bool)
    valid[0, :48] = False
    valid[1, 48:] = False
    return images, targets, valid


def test_cutmix_keeps_pixels_labels_and_review_masks_aligned_without_mutating_inputs():
    arrays = labelled_crops()
    originals = [array.copy() for array in arrays]
    mixed = cutmix_light_crops(*arrays, np.random.default_rng(5), probability=1.)
    repeated = cutmix_light_crops(*arrays, np.random.default_rng(5), probability=1.)
    images, targets, valid = mixed
    np.testing.assert_array_equal(images[:, 0, ::4, ::4], targets)
    donors = (images[:, 1, ::4, ::4] * 4).astype(int)
    yy, xx = np.indices(targets.shape[1:])
    inherited_valid = arrays[2][donors, yy, xx]
    assert not np.any(valid & ~inherited_valid)
    assert valid.any(axis=(1, 2)).all()
    assert targets.dtype == np.float32 and valid.dtype == bool
    assert images.min() >= 0 and images.max() <= 1
    for index, changed in enumerate(donors != np.arange(4)[:, None, None]):
        assert changed.any() and not changed.all()
        ys, xs = np.where(changed)
        # Both sides of every cut are unsupervised, including chopped centers.
        assert not valid[index, ys.min():ys.max() + 1, xs.min() - 1:xs.min() + 1].any()
        assert not valid[index, ys.min():ys.max() + 1, xs.max():xs.max() + 2].any()
        assert not valid[index, ys.min() - 1:ys.min() + 1, xs.min():xs.max() + 1].any()
        assert not valid[index, ys.max():ys.max() + 2, xs.min():xs.max() + 1].any()
    for original, current, result, same_result in zip(originals, arrays, mixed, repeated):
        np.testing.assert_array_equal(original, current)
        np.testing.assert_array_equal(result, same_result)
        assert not np.shares_memory(result, current)


def test_cutmix_disabled_preserves_arrays_and_random_stream():
    arrays = labelled_crops()
    rng = np.random.default_rng(13)
    result = cutmix_light_crops(*arrays, rng)
    assert all(actual is original for actual, original in zip(result, arrays))
    assert rng.random() == np.random.default_rng(13).random()
    # A skipped augmentation also avoids copies.
    result = cutmix_light_crops(*arrays, np.random.default_rng(13), probability=.01)
    assert all(actual is original for actual, original in zip(result, arrays))


def test_cutmix_retains_original_crops_if_the_cut_erases_all_reviewed_pixels():
    images, targets, _ = labelled_crops()
    dense = np.ones_like(targets, dtype=bool)
    _, _, mixed_valid = cutmix_light_crops(images, targets, dense, np.random.default_rng(5), 1.)
    y, x = np.argwhere(~mixed_valid[0])[0]
    sparse = np.zeros_like(dense)
    sparse[:, y, x] = True
    result = cutmix_light_crops(images, targets, sparse, np.random.default_rng(5), 1.)
    for actual, original in zip(result, (images, targets, sparse)):
        np.testing.assert_array_equal(actual, original)


def test_cutmix_rejects_invalid_probability_and_incompatible_batches():
    images, targets, valid = labelled_crops()
    rng = np.random.default_rng(5)
    for probability in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="probability"):
            cutmix_light_crops(images, targets, valid, rng, probability)
    with pytest.raises(ValueError, match="at least two"):
        cutmix_light_crops(images[:1], targets[:1], valid[:1], rng, 1.)
    with pytest.raises(ValueError, match="shapes must agree"):
        cutmix_light_crops(images[:, :, :-1], targets, valid, rng, 1.)
    with pytest.raises(ValueError, match="shapes must agree"):
        cutmix_light_crops(images, targets, valid[:, :-1], rng, 1.)
