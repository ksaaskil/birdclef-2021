from math import ceil
from pathlib import Path

import pytest

from src.dataset import (
    use_data_root,
    short_audio_metadata_ds,
    short_audio_ds,
    use_train_metadata_csv,
)
from src.spectrogram import MELS, STRIDE

RESOURCES = Path("tests") / "resources"


def test_metadata_dataset():

    with use_data_root(str(RESOURCES)):

        ds = short_audio_metadata_ds()

        for sample in ds.take(1):
            assert "primary_label" in sample


def test_audio_dataset():

    with use_data_root(str(RESOURCES)):

        ds = short_audio_ds()

        for sample, label in ds.take(2):
            assert sample["primary_label"] is not None
            assert sample["mel_spec"] is not None
            mel_spec = sample["mel_spec"]
            mel_spec_width = ceil(32000 * 5 / STRIDE)
            assert mel_spec.shape == (mel_spec_width, MELS)

            assert len(label) == 1
            assert label[0] == 1


@pytest.mark.skip(reason="Uses derived dataset")
def test_small_audio_dataset():

    with use_data_root("data"), use_train_metadata_csv("train_metadata_small.csv"):

        ds = short_audio_ds()

        for sample, label in ds.take(2).as_numpy_iterator():
            assert sample["primary_label"] is not None
            assert sample["mel_spec"] is not None
            mel_spec = sample["mel_spec"]
            mel_spec_width = ceil(32000 * 5 / STRIDE)
            assert mel_spec.shape == (mel_spec_width, MELS)
            assert len(label) == 27
