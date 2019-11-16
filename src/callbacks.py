import mlflow
from typing import Optional


class RecorderMlFlow(Callback):
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
