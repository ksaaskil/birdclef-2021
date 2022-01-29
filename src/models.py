from dataclasses import dataclass
import typing

from tensorflow import keras
from keras import layers

DEFAULT_MODEL = "smoke-test"

INPUT_SHAPE = (313, 128)


@dataclass(frozen=True)
class ModelBuilderConfig:
    n_classes: int


class ModelBuilder(typing.Protocol):
    def __call__(config: ModelBuilderConfig) -> keras.Model:
        ...


def build_smoke_test_model(config: ModelBuilderConfig):
    input_layer = keras.Input(shape=INPUT_SHAPE, name="mel_spectrogram")
    x = layers.Flatten()(input_layer)
    output = layers.Dense(config.n_classes, activation="softmax")(x)
    return keras.Model(inputs={"mel_spec": input_layer}, outputs=output)


MODELS: typing.Mapping[str, ModelBuilder] = {"smoke-test": build_smoke_test_model}


def get_model_builder(model: str) -> ModelBuilder:
    factory = MODELS[model]

    return factory
