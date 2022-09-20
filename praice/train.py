import datetime
import shutil
import time
from pathlib import Path, PosixPath
from typing import Tuple, Union

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn import metrics

from . import ml, paths, utils
from .tickers import Instrument

plt.style.use("ggplot")


class Trainer:
    def __init__(
        self,
        ticker: str,
        learners_cnf: str = "default_ml_config",
        datasets_cnf: str = "default_instrument_config",
        custom_learners_cnf_fp: Union[str, PosixPath] = None,
        custom_datasets_cnf_fp: Union[str, PosixPath] = None,
        experiment_name: str = None,
    ):
        if learners_cnf is None and custom_learners_cnf_fp is None:
            raise AssertionError(
                "One of the 'learners_cnf' or 'custom_learners_cnf_fp' should be provided."
            )

        if datasets_cnf is None and custom_datasets_cnf_fp is None:
            raise AssertionError(
                "One of the 'datasets_cnf' or 'custom_datasets_cnf_fp' should be provided."
            )

        self.ticker = ticker
        self.experiment_name = (
            experiment_name
            if experiment_name is not None
            else f"{datetime.datetime.utcnow().date()}__{ticker}"
        )
        self.learners_cnf_fp = (
            str(paths.ML_CONFIGS_DIR / f"{learners_cnf}.yaml")
            if custom_learners_cnf_fp is None
            else str(custom_learners_cnf_fp)
        )
        self.datasets_cnf_fp = (
            str(paths.INSTRUMENT_CONFIGS_DIR / f"{datasets_cnf}.yaml")
            if custom_datasets_cnf_fp is None
            else str(custom_datasets_cnf_fp)
        )

    @staticmethod
    def __get_instrument_datasets(
        ticker: str, data_params: dict
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Get train, validation, and test sets using given configs.

        Args:
            ticker (str): Ticker name.
            data_params (dict): Data configurations.

        Returns:
            Tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ]:
                X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train, X_val, X_test, y_train, y_val, y_test = Instrument(
            ticker=ticker,
            period=data_params["period"],
            add_ta_indicators=data_params["add_ta_indicators"],
            ta_indicators_config_fn=data_params["ta_indicators_config_fn"],
            add_date_features=data_params["add_date_features"],
            lookback_period=data_params["lookback_period"],
            add_past_close_prices=data_params["add_past_close_prices"],
            add_past_pct_changes=data_params["add_past_pct_changes"],
            forecast_period=data_params["forecast_period"],
        ).get_data(
            train_size=data_params["train_size"],
            val_size=data_params["val_size"],
            test_size=data_params["test_size"],
            separate_y=data_params["separate_y"],
            dropna=data_params["dropna"],
            environment=data_params["environment"],
        )
        return (X_train, X_val, X_test, y_train, y_val, y_test)

    @staticmethod
    def __fit_estimator(
        model: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        settings: dict,
    ) -> Tuple[ml.IEstimator, str]:
        """Create and train a ml.learner

        Args:
            model (str): Model name.
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training labels.
            settings (dict): Model hyper-parameters.

        Returns:
            Tuple[IEstimator, str]: (fitted model, fit time)
        """
        estimator = ml.learner(model=model)
        print(f"Training {estimator.__model__} estimator ...\n")
        t1 = time.perf_counter()
        estimator.fit(X_train=X_train, y_train=y_train, settings=settings)
        t2 = time.perf_counter()
        fit_time = str(utils.format_time(seconds=t2 - t1))
        return (estimator, fit_time)

    @staticmethod
    def __log_mlflow_metrics(
        experiment_id: str,
        run_id: str,
        y_true: Union[np.array, pd.DataFrame],
        y_pred: Union[np.array, pd.DataFrame],
        task: str = "regression",
    ):
        """Compute and log metrics of an mlflow run.

        Args:
            experiment_id (str): mlflow experiment id.
            run_id (str): mlflow run id.
            y_true (Union[np.array, pd.DataFrame]): Ground truth labels.
            y_pred (Union[np.array, pd.DataFrame]): Predicted labels.
            task (str, optional): ML problem. Valid tasks: classification, regression. Defaults to "regression".
        """
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id, nested=True
        ):
            if task == "regression":
                mlflow.log_metric(
                    "rmse",
                    metrics.mean_squared_error(y_true, y_pred, squared=False),
                )
                mlflow.log_metric(
                    "mae",
                    metrics.mean_absolute_error(y_true, y_pred),
                )
                mlflow.log_metric(
                    "r2",
                    metrics.r2_score(y_true, y_pred),
                )
                mlflow.log_metric(
                    "mape",
                    metrics.mean_absolute_percentage_error(y_true, y_pred),
                )
            elif task == "classification":
                mlflow.log_metric(
                    "accuracy",
                    metrics.accuracy_score(y_true, y_pred),
                )
                mlflow.log_metric(
                    "f1_micro",
                    metrics.f1_score(y_true, y_pred, average="micro"),
                )
                mlflow.log_metric(
                    "avg_precision_macro",
                    metrics.average_precision_score(
                        y_true, y_pred, average="macro"
                    ),
                )

    @staticmethod
    def __plot(
        y_true,
        y_pred,
        title: str = "",
        parent_dir: PosixPath = None,
        filename: str = None,
    ):
        y_df = pd.DataFrame(index=y_true.index)
        y_df["actual"] = y_true.values
        y_df["preds"] = y_pred
        plt.plot(y_df["actual"], label="Actual")
        plt.plot(y_df["preds"], label="Predicted")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend()
        fn = (
            f"{utils.unique_id()}.png"
            if filename is None
            else f"{filename}.png"
        )
        fp = fn if parent_dir is None else parent_dir / f"{fn}"
        plt.savefig(fp)
        plt.close()
        return fp

    @classmethod
    def __log_mlflow_artifacts(
        cls,
        experiment_id: str,
        run_id: str,
        estimator: ml.IEstimator,
        y_true: Union[np.array, pd.DataFrame],
        y_pred: Union[np.array, pd.DataFrame],
        plot_title: str = None,
        model_filename: str = "model",
        plot_filename: str = "val_plot",
        test_set: pd.DataFrame = None,
        test_filename: str = "test",
    ):
        parent_dir = Path(".").resolve() / f"{utils.unique_id()}"
        ml.save_estimator(
            estimator=estimator, parent_dir=parent_dir, filename=model_filename
        )
        cls.__plot(
            y_true=y_true,
            y_pred=y_pred,
            title=plot_title,
            parent_dir=parent_dir,
            filename=plot_filename,
        )
        if test_set is not None and len(test_set) > 0:
            test_set.to_csv(parent_dir / f"{test_filename}.csv", index=True)
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id, nested=True
        ):
            mlflow.log_artifacts(parent_dir)

        shutil.rmtree(parent_dir)

    def run(self, save_artifacts: bool = True):
        ml_config = utils.load_yaml(
            fp=self.learners_cnf_fp, validate=True, config_type="learners"
        )
        instrument_config = utils.load_yaml(
            fp=self.datasets_cnf_fp, validate=True, config_type="instruments"
        )

        exp = mlflow.set_experiment(self.experiment_name)
        for data_params in instrument_config["data"]:
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = self.__get_instrument_datasets(
                ticker=self.ticker, data_params=data_params
            )

            if len(X_val) == 0 and len(y_val) == 0:
                X_val = X_train
                y_val = y_train

            for learner in ml_config["learners"]:
                for ml_params in learner["settings"]:
                    # fit and predict
                    estimator, fit_time = self.__fit_estimator(
                        model=learner["model"],
                        X_train=X_train,
                        y_train=y_train,
                        settings=ml_params,
                    )
                    y_pred = estimator.predict(X_val)
                    if len(X_test) > 0:
                        test_set = X_test.copy()
                        test_set["prediction"] = estimator.predict(X_test)
                        test_set["forecast_period"] = data_params[
                            "forecast_period"
                        ]
                    # experiment tracking
                    with mlflow.start_run(
                        experiment_id=exp.experiment_id
                    ) as run:
                        run_id = run.info.run_id
                        # set experiment tags
                        mlflow.set_tag("ticker", self.ticker)
                        mlflow.set_tag("estimator", estimator.__model__)
                        mlflow.set_tag("fit_time", fit_time)
                        mlflow.set_tag("train_size", len(y_train))
                        mlflow.set_tag("val_size", len(y_val))
                        # log parameters and metrics
                        mlflow.log_params(dict(**data_params, **ml_params))
                        self.__log_mlflow_metrics(
                            experiment_id=exp.experiment_id,
                            run_id=run_id,
                            y_true=y_val,
                            y_pred=y_pred,
                            task=estimator.__task__,
                        )
                        # save experiment artifacts
                        if save_artifacts:
                            self.__log_mlflow_artifacts(
                                experiment_id=exp.experiment_id,
                                run_id=run_id,
                                estimator=estimator,
                                y_true=y_val,
                                y_pred=y_pred,
                                plot_title=self.ticker,
                                test_set=test_set,
                            )
