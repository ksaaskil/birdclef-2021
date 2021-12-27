# Bird call identification

See [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021#) Kaggle competition.

## Getting started

Install dependencies:

```bash
$ pip install -e '.[dev]'
```

### Setup `kaggle`

Sign in to Kaggle. Follow the [instructions](https://github.com/Kaggle/kaggle-api) to prepare `~/.kaggle/kaggle.json` file.

### Working with data

See the [Data](https://www.kaggle.com/c/birdclef-2021/data) page.

Download the 39 GiB dataset:

```bash
$ kaggle competitions download birdclef-2021 -p data
```

Download single file:

```bash
$ kaggle competitions download birdclef-2021 -p data/train_short_audio/acafly -f train_short_audio/acafly/XC109605.ogg
```

List all files in CSV format

```bash
$ kaggle competitions files birdclef-2021 --csv
```

### Setup Jupyter

Install kernel:

```bash
$ python -m ipykernel install --user --name bird-3.8.1 --display-name "Python (bird-3.8.1)"
```