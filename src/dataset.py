import logging
import importlib
import typing

import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import src.preprocess

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

def read_classes() -> typing.Sequence[str]:
    return sorted(l.decode() for l in train_metadata_ds().map(lambda x: x["primary_label"]).unique().as_numpy_iterator())

def primary_label_to_tensor(primary_label: str, classes: typing.Sequence[str]) -> tf.Tensor:
    return tf.cast(classes == primary_label, tf.int32)

def tensor_to_class(label: tf.Tensor, classes: typing.Sequence[str]) -> str:
    label_index = np.argmax(label.numpy())
    return classes[label_index]

