"""Download files matching given metadata dataset."""
import logging
from pathlib import Path
import subprocess

from src.dataset import use_data_root, use_train_metadata_csv, short_audio_metadata_ds

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def exec_cmd(cmd):
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception(f"Command failed: {cmd}")


def gs_copy(src: str, dst: str):
    cmd = ["gsutil", "-m", "cp", src, dst]
    exec_cmd(cmd)


SOURCE_DIR = "gs://bird-clef-v2/data"


def main():
    # TODO

    data_root = "data"
    with use_data_root("data"), use_train_metadata_csv("train_metadata_small.csv"):
        ds = short_audio_metadata_ds()

        for sample in ds.as_numpy_iterator():
            filename = f"train_short_audio/{sample['primary_label'].decode()}/{sample['filename'].decode()}"
            src = f"{SOURCE_DIR}/{filename}"
            dst = Path(data_root).joinpath(filename).resolve()
            logger.info(f"Copying from {src} to {dst}")
            gs_copy(src, str(dst))


if __name__ == "__main__":
    main()
