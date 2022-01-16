DEFAULT_MODEL = "baseline"


def baseline_model():
    return


MODELS = {"baseline": baseline_model}


def get_model(model: str):
    factory = MODELS[model]

    return factory()
