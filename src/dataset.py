import contextlib
import logging
import os
import typing

import tensorflow_io as tfio
import tensorflow as tf
import numpy as np

from src.spectrogram import compute_mel_spectrogram

_DATA_ROOT = os.getenv("DATA_ROOT", "gs://bird-clef-v2/data")
_TRAIN_METADATA_CSV = "train_metadata.csv"

SR = 32000
SPLIT_SECS = 5

BATCH_SIZE = 32


def data_root() -> str:
    return _DATA_ROOT


def train_metadata_csv() -> str:
    return _TRAIN_METADATA_CSV


def train_short_audio_data() -> str:
    return f"{data_root()}/train_short_audio"


@contextlib.contextmanager
def use_data_root(root: str):
    global _DATA_ROOT
    original = _DATA_ROOT

    try:
        _DATA_ROOT = root
        yield
    finally:
        _DATA_ROOT = original


@contextlib.contextmanager
def use_train_metadata_csv(filename: str):
    global _TRAIN_METADATA_CSV
    original = _TRAIN_METADATA_CSV

    try:
        _TRAIN_METADATA_CSV = filename
        yield
    finally:
        _TRAIN_METADATA_CSV = original


def short_audio_metadata_csv() -> str:
    return f"{data_root()}/{train_metadata_csv()}"


COLUMNS = [
    "primary_label",
    "secondary_labels",
    "type",
    "latitude",
    "longitude",
    "scientific_name",
    "common_name",
    "date",
    "filename",
    "rating",
    "time",
]


def short_audio_metadata_ds() -> tf.data.Dataset:
    def squeeze(row):
        return {key: tf.squeeze(value, axis=0) for key, value in row.items()}

    def add_file_path(row):
        row = {**row, "file_path": get_file_path(row)}
        return row

    return (
        tf.data.experimental.make_csv_dataset(
            short_audio_metadata_csv(),
            batch_size=1,
            select_columns=COLUMNS,
            num_epochs=1
            # ).flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) # Generate_scalars
        )
        .map(squeeze)
        .map(add_file_path)
    )


def classes_(metadata_ds: tf.data.Dataset) -> typing.Sequence[str]:
    return sorted(
        l.decode()
        for l in metadata_ds.map(lambda x: x["primary_label"])
        .unique()
        .as_numpy_iterator()
    )


def read_classes() -> typing.Sequence[str]:
    metadata_ds = short_audio_metadata_ds()
    return classes_(metadata_ds)


def primary_label_to_tensor(
    primary_label: str, classes: typing.Sequence[str]
) -> tf.Tensor:
    return tf.cast(classes == primary_label, tf.int32)


def tensor_to_class(label: tf.Tensor, classes: typing.Sequence[str]) -> str:
    label_index = np.argmax(label.numpy())
    return classes[label_index]


def read_file(url) -> tf.Tensor:
    logging.debug(f"Reading file: {url}")
    return tf.squeeze(
        tfio.audio.AudioIOTensor(url).to_tensor(), axis=1
    )  # remove channel axis


def get_file_path(row) -> str:
    """Full path to audio file."""
    filename = row["filename"]
    primary_label = row["primary_label"]
    file_url = train_short_audio_data() + "/" + primary_label + "/" + filename
    return file_url


def add_audio(row):
    file_url = get_file_path(row)
    [
        audio,
    ] = tf.py_function(read_file, [file_url], [tf.float32])
    return {**row, "file_url": file_url, "audio": audio}


def split_audio(audio: tf.Tensor) -> tf.Tensor:
    length = audio.shape[-1]

    n_splits = length // (SR * SPLIT_SECS)

    splits = []

    for i in range(n_splits):
        start = i * SR * SPLIT_SECS
        end = (i + 1) * SR * SPLIT_SECS
        split = audio[start:end]
        splits.append(split)

    return tf.convert_to_tensor(splits)


TensorMap = typing.Mapping[str, tf.Tensor]


def split_to_segments(sample: TensorMap, label: tf.Tensor) -> tf.data.Dataset:
    audio = sample["audio"]

    splits = tf.py_function(split_audio, [audio], tf.float32)

    return tf.data.Dataset.from_tensor_slices(
        {"segment": splits, "segment_i": tf.range(tf.shape(splits)[0])}
    ).map(lambda x: ({**sample, **x}, label))


def add_label(classes: typing.Sequence[str]):
    def add_(sample: TensorMap) -> typing.Tuple[TensorMap, tf.Tensor]:
        label = primary_label_to_tensor(sample["primary_label"], classes)
        return sample, label

    return add_


def drop_keys(*keys):
    def drop(sample, label):
        sample = sample.copy()
        for key in keys:
            sample.pop(key)
        return sample, label

    return drop


def add_spectrogram(sample, label):
    sample["mel_spec"] = compute_mel_spectrogram(sample["segment"])
    return sample, label


def add_audio_fn(
    sample: TensorMap, label: tf.Tensor
) -> typing.Tuple[TensorMap, tf.Tensor]:
    return add_audio(sample), label


def add_fold(buckets=5):
    def add_(sample, label):
        fold = tf.strings.to_hash_bucket(sample["filename"], num_buckets=buckets)
        return {**sample, "fold": fold}, label

    return add_


def short_audio_ds() -> tf.data.Dataset:
    metadata_ds = short_audio_metadata_ds()
    classes = classes_(metadata_ds)
    return (
        metadata_ds.map(add_label(classes))
        .map(add_fold())
        .map(add_audio_fn)
        .flat_map(split_to_segments)
        .map(add_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
        .map(drop_keys("audio"), num_parallel_calls=tf.data.AUTOTUNE)
    )


def configure_for_training(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
