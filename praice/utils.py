import datetime
import os
import uuid
from pathlib import PosixPath
from typing import List, Union

import yaml

from . import ml


def validate_instruments_config(config: dict):
    """Validate instruments yaml config file.

    Args:
        config (dict): Loaded yaml.

    Raises:
        AssertionError: If something is wrong in config file.

    Returns:
        bool: Returns True if the config file is in proper format.
    """
    assert (
        "data" in config.keys()
    ), "Instruments config file should contain a 'data' key."
    forecast_periods = []
    for o in config["data"]:
        assert (
            "period" in o.keys()
        ), "Each data item should contain 'period' key."
        assert (
            "add_ta_indicators" in o.keys()
        ), "Each data item should contain 'add_ta_indicators' key."
        assert (
            "ta_indicators_config_fn" in o.keys()
        ), "Each data item should contain 'ta_indicators_config_fn' key."
        assert (
            "add_date_features" in o.keys()
        ), "Each data item should contain 'add_date_features' key."
        assert (
            "lookback_period" in o.keys()
        ), "Each data item should contain 'lookback_period' key."
        assert (
            "add_past_close_prices" in o.keys()
        ), "Each data item should contain 'add_past_close_prices' key."
        assert (
            "add_past_pct_changes" in o.keys()
        ), "Each data item should contain 'add_past_pct_changes' key."
        assert (
            "forecast_period" in o.keys()
        ), "Each data item should contain 'forecast_period' key."
        assert (
            "train_size" in o.keys()
        ), "Each data item should contain 'train_size' key."
        assert (
            "val_size" in o.keys()
        ), "Each data item should contain 'val_size' key."
        assert (
            "test_size" in o.keys()
        ), "Each data item should contain 'test_size' key."
        assert (
            "separate_y" in o.keys()
        ), "Each data item should contain 'separate_y' key."
        assert (
            "dropna" in o.keys()
        ), "Each data item should contain 'dropna' key."
        assert (
            "environment" in o.keys()
        ), "Each data item should contain 'environment' key."
        forecast_periods.append(o["forecast_period"])

    assert (
        len(set(forecast_periods)) == 1
    ), "You can not have different values of 'forecast_period' in one instrument config file."
    return True


def validate_learners_config(config: dict):
    """Validate learners yaml config file.

    Args:
        config (dict): Loaded yaml.

    Raises:
        AssertionError: If something is wrong in config file.

    Returns:
        bool: Returns True if the config file is in proper format.
    """
    assert (
        "learners" in config.keys()
    ), "learners config should contain 'learners' key."
    learners: List[dict] = config["learners"]
    for model in learners:
        assert (
            "model" in model.keys()
        ), "Each item below 'learners' should contain a 'model' key."
        estimator_ = ml.learner(model["model"])
        assert (
            "settings" in model.keys()
        ), "Each item below 'learners' should contain a 'settings' key."
        for setting in model["settings"]:
            assert set(setting.keys()).issubset(estimator_.valid_params), (
                f"Provided settings ({list(setting.keys())}) is not a subset of {model['model']} "
                f"valid parameters ({estimator_.valid_params})"
            )
    return True


def validate_ta_indicators_config(config: dict):
    """Validate ta_indicators yaml config file.

    Args:
        config (dict): Loaded yaml.

    Raises:
        AssertionError: If something is wrong in config file.

    Returns:
        bool: Returns True if the config file is in proper format.
    """
    # TODO: validate ta_indicators config file
    return True


def load_yaml(
    fp: Union[str, PosixPath], validate: bool = False, config_type: str = None
):
    """Load a yaml file from disk.

    Args:
        fp (Union[str, PosixPath]): File path to yaml file.
        validate (bool, optional): If True, validate config file to make sure it
            has the correct format. Defaults to False.
        config_type (str, optional): Type of yaml config file. Valid config types:
            instruments, learners, ta_indicators. Defaults to None.

    Raises:
        FileNotFoundError: If file path is incorrect.

    Returns:
        dict: Loaded yaml.
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"There is no such a file named {fp}")
    with open(str(fp), "r") as f:
        config = yaml.safe_load(f)

    if validate:
        CONFIG_VALIDATORS = {
            "instruments": validate_instruments_config,
            "learners": validate_learners_config,
            "ta_indicators": validate_ta_indicators_config,
        }
        assert (
            config_type is not None and config_type in CONFIG_VALIDATORS.keys()
        ), f"'config_type' should be one of {list(CONFIG_VALIDATORS.keys())} when 'validate' is True"
        CONFIG_VALIDATORS[config_type](config=config)

    return config


def format_time(seconds: Union[int, float]):
    """Convert seconds to hh:mm:ss representation.

    Args:
        seconds (Union[int, float]): Time in seconds.

    Returns:
        str: Time in hh:mm:ss format.
    """
    return str(datetime.timedelta(seconds=int(seconds)))


def unique_id():
    """Generate a random unique ID.

    Returns:
        str: Unique ID
    """
    return uuid.uuid4().hex
