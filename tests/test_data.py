from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mnist_benchmark.data import (
    DatasetValidationError,
    augment_with_translations,
    load_mnist_csv,
    split_dataset,
    translate_images,
)


def test_translate_images_moves_pixel_right() -> None:
    images = np.zeros((1, 784), dtype=np.uint8)
    images[0, 0] = 255

    translated = translate_images(images, dx=1, dy=0).reshape(28, 28)

    assert translated[0, 0] == 0
    assert translated[0, 1] == 255


def test_augmentation_preserves_label_blocks() -> None:
    images = np.zeros((2, 784), dtype=np.uint8)
    labels = np.array([3, 7])

    augmented_images, augmented_labels = augment_with_translations(
        images,
        labels,
        shifts=((1, 0),),
    )

    assert augmented_images.shape == (4, 784)
    np.testing.assert_array_equal(augmented_labels, [3, 7, 3, 7])


def test_loader_rejects_wrong_column_count(tmp_path: Path) -> None:
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame([[1, 0, 255]]).to_csv(invalid_csv, header=False, index=False)

    with pytest.raises(DatasetValidationError, match="785 columns"):
        load_mnist_csv(invalid_csv)


def test_split_rejects_sample_too_small_for_all_classes() -> None:
    features = np.zeros((20, 784), dtype=np.float32)
    labels = np.tile(np.arange(10), 2)
    feature_names = tuple(f"pixel_{index:03d}" for index in range(784))

    with pytest.raises(ValueError, match="too small"):
        split_dataset(
            features,
            labels,
            feature_names,
            test_size=0.3,
            random_state=42,
        )
