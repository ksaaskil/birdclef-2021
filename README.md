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

Download `train_metadata.csv`:

```bash
$ kaggle competitions download birdclef-2021 -p data -f train_metadata.csv
$ unzip data/train_metadata.csv.zip -d data
```

### Setup Jupyter

Install kernel:

```bash
$ python -m ipykernel install --user --name bird-3.8.1 --display-name "Python (bird-3.8.1)"
```

## Resources

- [Overview of BirdCLEF 2021](http://ceur-ws.org/Vol-2936/paper-123.pdf)
- [Where to start: A collection of resources](https://www.kaggle.com/c/birdclef-2021/discussion/230000)
- [Best working note awards](https://www.kaggle.com/c/birdclef-2021/discussion/252995)

## Tips

### Downloading data

Create a GCP VM with at least 100 GB disk space. Give write access to Google Storage API.

SSH to the instance using:

```bash
$ gcloud compute ssh INSTANCE_NAME
```

Install `pip`, `tmux` and `unzip`:

```bash
$ sudo apt install python3-pip tmux unzip
```

Install Kaggle CLI:

```bash
$ pip3 install kaggle
```

Make directory `.kaggle` and transfer `kaggle.json` from your machine:

```bash
$ scp ~/.kaggle/kaggle.json USERNAME@VM_IP:~/.kaggle/kaggle.json
```

Create new tmux session:

```bash
$ tmux new -s kimmo
```

Download data to folder `data/`:

```bash
$ chmod 600 ~/.kaggle/kaggle.json
$ ./.local/bin/kaggle competitions download birdclef-2021 -p data
```

Detach from the session with `Ctrl+b d` and attach with `tmux a -t kimmo`.

Extract and copy data to Google bucket `bird-clef-kimmo`:

```bash
$ unzip data/birdclef-2021.zip -d data
$ gsutil -m rsync -r data gs://bird-clef-kimmo/data
```

List and stop instances:

```bash
$ gcloud compute instances list
$ gcloud compute instances stop INSTANCE_NAME
```
