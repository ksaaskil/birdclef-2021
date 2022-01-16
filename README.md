# Bird call identification

See [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021#) Kaggle competition.

## Getting started

Install dependencies:

```bash
$ pip install -e '.[dev]'
```

On Linux, you also need to run `sudo apt-get install libsndfile1`.

Install also TensorFlow if not installed in your environment:

```bash
$ pip install -e .[tf]
```

### Pull data files

```bash
$ dvc pull
```

### Create down-sampled dataset

Create smaller dataset in `data/` folder:

```bash
$ python -m src.sample
```

### Download subset of files from Google Storage bucket

```bash
$ python -m src.download
```

### Train model

```bash
$ python -m src.train --model baseline --data-dir data
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
- [EfficientNet explained](https://paperswithcode.com/method/efficientnet)
- [LifeCLEF2022](https://www.imageclef.org/LifeCLEF2022)
- [freefield1010 dataset](https://arxiv.org/pdf/1309.5275.pdf)
- [Birdcall Identification Using CNN and Gradient Boosting Decision Trees with Weak and Noisy Supervision](http://ceur-ws.org/Vol-2936/paper-136.pdf)
- [Winning solution of BirdCLEF2021](https://github.com/namakemono/kaggle-birdclef-2021)
- [Bird audio detection challenge](http://dcase.community/challenge2018/task-bird-audio-detection)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)
- [TensorFlow IO Audio](https://www.tensorflow.org/io/tutorials/audio)
- [Simple Audio Recognition with TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio)

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

### Setting up a development instance in Vertex AI

Create a user-managed notebook in Vertex AI Workbench.

SSH to the instance with `jupyter` username:

```bash
$ gcloud compute ssh jupyter@bird-explore
```

Setup SSH configuration:

```bash
$ gcloud compute config-ssh
```

Switch `User` in `~/.ssh/config`:

```
# ~/.ssh/config
Host some-host
  User jupyter
```

Connecting from VS Code using the SSH host should now use `jupyter` as user, allowing you to use `/home/jupyter` for files and save remotely.

You can also setup port forwarding to `localhost` with:

```bash
$ gcloud compute ssh jupyter@bird-explore -- -N -L 8080:localhost:8080
```

### Setting up `kaggle`

Sign in to Kaggle. Follow the [instructions](https://github.com/Kaggle/kaggle-api) to prepare `~/.kaggle/kaggle.json` file.

### Working with data

See the [Data](https://www.kaggle.com/c/birdclef-2021/data) page.

Download the full 39 GiB dataset:

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