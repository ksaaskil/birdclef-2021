import argparse
import contextlib
import logging

from src.models import DEFAULT_MODEL, get_model
from src.dataset import use_data_root, short_audio_ds, configure_for_training


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model_name: str):
    logger.info(f"Training model: {model_name}")
    model = get_model(model_name)

    dataset = short_audio_ds()
    dataset = configure_for_training(dataset)
    # model.fit(dataset)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=None)

    return parser.parse_args()


def main():
    args = _parse_args()
    model = args.model
    data_dir = args.data_dir

    data_ctx = use_data_root(data_dir) if data_dir else contextlib.nullcontext()

    with data_ctx:
        train(model)


if __name__ == "__main__":
    main()
