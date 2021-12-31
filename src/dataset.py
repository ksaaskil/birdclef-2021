import logging
import importlib
import typing

import tensorflow_io as tfio
import tensorflow as tf
import numpy as np

import src.preprocess
from src.spectrogram import compute_mel_spectrogram

DATA_ROOT ="gs://bird-clef-kimmo/data"
TRAIN_SHORT_AUDIO_DATA = f"{DATA_ROOT}/train_short_audio"

SR = 32000
SPLIT_SECS = 5

def short_audio_metadata_csv(data_root: str) -> str:
    return f"{data_root}/train_metadata.csv"

COLUMNS = [  "primary_label","secondary_labels","type","latitude","longitude","scientific_name","common_name","date","filename","rating","time"
]

def short_audio_metadata_ds(data_root=DATA_ROOT) -> tf.data.Dataset:
    def squeeze(row):
        return { key: tf.squeeze(value, axis=0) for key, value in row.items() }

    return tf.data.experimental.make_csv_dataset(
        short_audio_metadata_csv(data_root),
        batch_size=4,
        select_columns=COLUMNS,
        num_epochs=1
    ).flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) # Generate_scalars

def classes_(metadata_ds: tf.data.Dataset) -> typing.Sequence[str]:
    return sorted(l.decode() for l in metadata_ds.map(lambda x: x["primary_label"]).unique().as_numpy_iterator())

def read_classes() -> typing.Sequence[str]:
    metadata_ds = short_audio_metadata_ds()
    return classes_(metadata_ds)

def primary_label_to_tensor(primary_label: str, classes: typing.Sequence[str]) -> tf.Tensor:
    return tf.cast(classes == primary_label, tf.int32)

def tensor_to_class(label: tf.Tensor, classes: typing.Sequence[str]) -> str:
    label_index = np.argmax(label.numpy())
    return classes[label_index]

def read_file(url) -> tf.Tensor:
    logging.debug(f"Reading file: {url}")
    return tf.squeeze(tfio.audio.AudioIOTensor(url).to_tensor(), axis=1)  # remove channel axis

def add_audio(row):
    filename = row["filename"]
    primary_label = row["primary_label"]
    file_url = TRAIN_SHORT_AUDIO_DATA + "/" + primary_label + "/" + filename
    [audio,] = tf.py_function(read_file, [file_url], [tf.float32])
    return {**row, "file_url": file_url, "audio": audio}

def split_audio(audio: tf.Tensor) -> tf.Tensor:
    length = audio.shape[-1]
    
    n_splits = length // (SR * SPLIT_SECS)
        
    splits = []
    
    for i in range(n_splits):
        start = i * SR*SPLIT_SECS
        end = (i+1) * SR*SPLIT_SECS
        split = audio[start:end]
        splits.append(split)
    
    return tf.convert_to_tensor(splits)

TensorMap = typing.Mapping[str, tf.Tensor]

def split_to_segments(sample: TensorMap, label: tf.Tensor) -> tf.data.Dataset:
    audio = sample["audio"]
    
    splits = tf.py_function(split_audio, [audio], tf.float32)
    
    return tf.data.Dataset.from_tensor_slices({"segment": splits}).map(lambda x: ({**sample, **x}, label))


def add_label(classes: typing.Sequence[str]):
    def add_(sample: TensorMap) -> typing.Tuple[TensorMap, tf.Tensor]:
        label = primary_label_to_tensor(sample["primary_label"], classes)
        return sample, label
    
    return add_

def drop_keys(*keys):
    def drop(rows):
        rows = rows.copy()
        for key in keys:
            rows.pop(key)
        return rows
    return drop

def add_spectrogram(sample, label):
    sample["mel_spec"] = compute_mel_spectrogram(sample["segment"])
    return sample, label

def add_audio_fn(sample: TensorMap, label: tf.Tensor) -> typing.Tuple[TensorMap, tf.Tensor]:
    return add_audio(sample), label

def short_audio_ds(data_root=DATA_ROOT) -> tf.data.Dataset:
    metadata_ds = short_audio_metadata_ds(data_root=data_root)
    classes = classes_(metadata_ds)
    return metadata_ds.map(add_label(classes)).map(add_audio_fn).flat_map(split_to_segments).map(add_spectrogram)
