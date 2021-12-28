from pathlib import Path
import typing

import pandas as pd

DATA_FOLDER = Path("data")


def add_country(df):
    import reverse_geocoder as rg
    latlng = df[["latitude", "longitude"]].apply(tuple, axis=1)
    places = rg.search(list(latlng))
    countries = [place["cc"] for place in places]
    return df.assign(country=countries)


def filter_by_species_within_countries(
    df: pd.DataFrame, countries: typing.Tuple[str] = ("FI",)
):
    with_country = add_country(df)

    sightings_within = with_country[with_country["country"].isin(countries)]

    species = set(sightings_within["scientific_name"].values)

    filtered = df[df["scientific_name"].isin(species)]

    assert len(filtered) > 0

    return filtered


def sample_small():
    df = pd.read_csv(DATA_FOLDER.joinpath("train_metadata.csv"))

    filtered = filter_by_species_within_countries(df)

    out = DATA_FOLDER.joinpath("sampled").joinpath("train_metadata.csv")

    out.parent.mkdir(exist_ok=True)

    filtered.to_csv(DATA_FOLDER.joinpath("sampled").joinpath("train_metadata.csv"))


if __name__ == "__main__":
    sample_small()
