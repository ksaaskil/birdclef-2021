"""
Download files one by one.
"""

from pathlib import Path
import subprocess
import time

import pandas as pd

DATA_FOLDER = Path("data")
COMPETITION_NAME = "birdclef-2021"


def exec_cmd(cmd):
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception(f"Command failed: {cmd}")


def download_one(primary_label: str, filename: str):

    target_folder = DATA_FOLDER.joinpath("train_short_audio").joinpath(primary_label)
    target_file = target_folder.joinpath(filename)

    if Path(target_file).exists():
        print(f"File exists, skipping downloading: {target_file}")
        pass

    cmd = [
        "kaggle",
        "competitions",
        "download",
        COMPETITION_NAME,
        "-p",
        str(target_folder),
        "-f",
        f"train_short_audio/{primary_label}/{filename}",
    ]

    exec_cmd(cmd)

    if not target_file.exists():
        zip_file = Path(f"{target_file}.zip")

        if not zip_file.exists():
            raise Exception(f"Expected to exist: {zip_file}")

        cmd = ["unzip", str(zip_file), "-d", str(zip_file.parent)]
        exec_cmd(cmd)

        zip_file.unlink()

    assert target_file.exists(), f"Expected to exist: {target_file}"


def download(metadata_file: Path):
    df = pd.read_csv(metadata_file)

    primary_label_and_filename = list(
        df[["primary_label", "filename"]].apply(tuple, axis=1)
    )

    for row in primary_label_and_filename:
        download_one(*row)
        time.sleep(1)


if __name__ == "__main__":
    download(
        metadata_file=DATA_FOLDER.joinpath("sampled").joinpath("train_metadata.csv")
    )
