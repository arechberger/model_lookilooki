import mlflow
from typing import Optional
from exp.nb_04 import Callback, AvgStatsCallback, listify


class RecorderMlFlowCallback(Callback):
    def __init__(
        self,
        experiment_name: str,
        run_name: str = "",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.step = 0

    def begin_fit(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def after_batch(self):
        if not self.in_train:
            return
        mlflow.log_metric("lr", self.opt.param_groups[-1]["lr"], step=self.step)
        mlflow.log_metric("loss", float(self.loss.detach().cpu()), step=self.step)
        self.step += 1

    def after_fit(self):
        mlflow.end_run()


class AvgStatsMlFlowCallback(AvgStatsCallback):
    def __init__(
        self,
        metrics,
        experiment_name: str,
        run_name: str = "",
        params: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
    ):
        super().__init__(metrics)
        self.metric_names = ["loss"] + [m.__name__ for m in listify(metrics)]
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.step = 0
        self.params = params

    def begin_fit(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        print(f"mlflow tracking uri: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        if isinstance(self.params, dict):
            for key, val in self.params.items():
                mlflow.log_param(key, val)

    def after_epoch(self):
        super().after_epoch()
        self._log_metrics(self.train_stats.avg_stats, "train")
        self._log_metrics(self.valid_stats.avg_stats, "valid")
        self.step += 1

    def _log_metrics(self, stats, prefix: str):
        for name, val in zip(self.metric_names, stats):
            mlflow.log_metric(f"{prefix}_{name}", float(val), self.step)

    def after_fit(self):
        mlflow.end_run()