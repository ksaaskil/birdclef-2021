from src.dataset import use_data_root
from pathlib import Path

from src.train import TrainingConfig, train

RESOURCES = Path("tests") / "resources"


def test_train():
    with use_data_root(str(RESOURCES)):
        config = TrainingConfig(epochs=2)
        train("smoke-test", config=config)
