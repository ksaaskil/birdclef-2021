from pathlib import Path

from src.dataset import use_data_root, short_audio_metadata_ds, short_audio_ds

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
