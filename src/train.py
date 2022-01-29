import argparse
import contextlib
from dataclasses import dataclass
import logging

from tensorflow.keras import optimizers, losses, metrics

from src.models import DEFAULT_MODEL, get_model_builder, ModelBuilderConfig
from src.dataset import (
    read_classes,
    use_data_root,
    short_audio_ds,
    configure_for_training,
)


logger = logging.getLogger(__name__)


def compile(model):
    model.compile(
        optimizer=optimizers.RMSprop(),
        # labels are one-hot encoded so SparseCategoricalCrossentropy does not apply
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 64


def train(model_name: str, config: TrainingConfig):
    logger.info(f"Training model: {model_name}")

    classes = read_classes()

    model_builder = get_model_builder(model_name)

    dataset = short_audio_ds()
    dataset = configure_for_training(dataset)

    model = model_builder(ModelBuilderConfig(n_classes=len(classes)))

    logger.info("Model summary")
    model.summary()

    compile(model)

    model.fit(dataset, epochs=config.epochs, batch_size=config.batch_size)


DEFAULT_DATA_DIR = "tests/resources"


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()
    model = args.model
    data_dir = args.data_dir

    data_ctx = use_data_root(data_dir) if data_dir else contextlib.nullcontext()

    config = TrainingConfig()

    with data_ctx:
        train(model, config)


if __name__ == "__main__":
    main()
