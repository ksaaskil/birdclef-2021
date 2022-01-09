"""Create smaller down-sampled dataset."""
from pathlib import Path

import pandas as pd

from src.dataset import short_audio_metadata_csv, use_data_root
from src.preprocess import add_country


def read_csv() -> pd.DataFrame:
    return pd.read_csv(short_audio_metadata_csv(), header=0)


COUNTRIES = ("FI",)


def main():
    with use_data_root("data/"):
        df = read_csv()
        print(f"Read {len(df)} rows")

        df = add_country(df)

        sightings_within = df[df["country"].isin(COUNTRIES)]
        species = set(sightings_within["scientific_name"].values)
        filtered = df[df["scientific_name"].isin(species)]

        print(f"Filtered to {len(filtered)} rows")

        output = Path("data") / "train_metadata_small.csv"

        print(f"Writing to file: {output}")
        filtered.to_csv(output, index=False)

if __name__ == "__main__":
    main()
