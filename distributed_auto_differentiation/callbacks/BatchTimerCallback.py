from catalyst.core.callback import Callback, CallbackOrder

from catalyst.callbacks.misc import TimerCallback

EPS = 1e-8

class BatchTimerCallback(TimerCallback):
    """Logs pipeline execution time."""
    def on_batch_end(self, runner: "IRunner") -> None:
        """Batch end hook.

        Args:
            runner: current runner
        """
        self.timer.stop("_timer/model_time")
        self.timer.stop("_timer/batch_time")

        # @TODO: just a trick
        self.timer.elapsed["_timer/_fps"] = runner.batch_size / (
            self.timer.elapsed["_timer/batch_time"] + EPS
        )
        for key, value in self.timer.elapsed.items():
            runner.batch_metrics[key] = value
        if hasattr(runner, "loader_metrics"):
            for key, value in self.timer.elapsed.items():
                if key not in runner.loader_metrics.keys():
                    runner.loader_metrics[key] = value
                else:
                    runner.loader_metrics[key] += value
                

        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")