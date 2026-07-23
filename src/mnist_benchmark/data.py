"""Dataset loading, validation, splitting, and augmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_SIDE = 28
FEATURE_COUNT = IMAGE_SIDE * IMAGE_SIDE
EXPECTED_COLUMN_COUNT = FEATURE_COUNT + 1
VALID_LABELS = set(range(10))


class DatasetValidationError(ValueError):
    """Raised when a CSV file does not follow the expected MNIST schema."""


@dataclass(frozen=True)
class DatasetSplit:
    """A stratified train/test split."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def load_mnist_csv(
    path: Path,
    *,
    sample_limit: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load and validate a headerless MNIST CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        frame = pd.read_csv(path, header=None)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DatasetValidationError(f"The CSV file could not be parsed: {error}") from error
    if frame.empty:
        raise DatasetValidationError("The dataset is empty.")
    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {frame.shape[1]}."
        )

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise DatasetValidationError("The dataset contains missing or non-numeric values.")

    labels = numeric_frame.iloc[:, 0].to_numpy()
    if not np.equal(labels, np.floor(labels)).all():
        raise DatasetValidationError("Class labels must be integers.")
    labels = labels.astype(np.int64)

    unknown_labels = set(np.unique(labels)) - VALID_LABELS
    if unknown_labels:
        raise DatasetValidationError(
            f"Labels must be in the 0-9 range; found {sorted(unknown_labels)}."
        )

    pixels = numeric_frame.iloc[:, 1:]
    pixel_min = float(pixels.min().min())
    pixel_max = float(pixels.max().max())
    if pixel_min < 0 or pixel_max > 255:
        raise DatasetValidationError(
            f"Pixel values must be between 0 and 255; found {pixel_min:g} to {pixel_max:g}."
        )

    if sample_limit is not None:
        if sample_limit < len(VALID_LABELS) * 2:
            raise ValueError("sample_limit must be at least 20 for a stratified split.")
        if sample_limit < len(numeric_frame):
            numeric_frame, _ = train_test_split(
                numeric_frame,
                train_size=sample_limit,
                stratify=labels,
                random_state=random_state,
            )
            labels = numeric_frame.iloc[:, 0].to_numpy(dtype=np.int64)
            pixels = numeric_frame.iloc[:, 1:]

    features = pixels.to_numpy(dtype=np.float32)
    feature_names = tuple(f"pixel_{index:03d}" for index in range(FEATURE_COUNT))
    return features, labels, feature_names


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    """Create a reproducible stratified split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    class_count = len(np.unique(labels))
    test_count = int(np.ceil(len(labels) * test_size))
    training_count = len(labels) - test_count
    if test_count < class_count or training_count < class_count:
        raise ValueError(
            "The selected sample is too small to place every class in both splits."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test, feature_names)


def translate_images(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a batch of flattened images and fill exposed pixels with zero."""
    if images.ndim != 2 or images.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected an (n, {FEATURE_COUNT}) image matrix.")
    if abs(dx) >= IMAGE_SIDE or abs(dy) >= IMAGE_SIDE:
        raise ValueError("The translation must be smaller than the image dimensions.")

    image_batch = images.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    translated = np.zeros_like(image_batch)

    source_x, target_x = _translation_slices(IMAGE_SIDE, dx)
    source_y, target_y = _translation_slices(IMAGE_SIDE, dy)
    translated[:, target_y, target_x] = image_batch[:, source_y, source_x]
    return translated.reshape(-1, FEATURE_COUNT)


def augment_with_translations(
    images: np.ndarray,
    labels: np.ndarray,
    shifts: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1)),
) -> tuple[np.ndarray, np.ndarray]:
    """Add translated copies of each training image."""
    if len(images) != len(labels):
        raise ValueError("The image and label counts do not match.")

    augmented_images = [images]
    for dx, dy in shifts:
        augmented_images.append(translate_images(images, dx, dy))

    return np.concatenate(augmented_images), np.tile(labels, len(augmented_images))


def _translation_slices(length: int, offset: int) -> tuple[slice, slice]:
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)
