from functools import lru_cache
from typing import Iterable
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn import metrics as skl_metrics
import torchvision
import os
from pathlib import Path
import pandas as pd


class TrainingMetric:
    def __init__(self, metric_func, metric_name, optimum=None):
        self.func = metric_func
        self.name = metric_name
        self.optimum = optimum

    def calc_metric(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except ValueError as e:
            return np.nan

    def __call__(self, y_true, y_pred, labels=None, split=None, step_type=None) -> dict:

        # if y_true is empty
        if y_true.shape[0] == 0:  # TODO: handle other cases
            m = {
                f"{step_type}_{split}_{l}_{self.name}": self.calc_metric(None, yp)
                for yp, l in zip(y_pred.T, labels)
            }
            return m

        # Simple 1:1 y_true and y_pred are either shape=(batch, 1) or shape=(batch,)
        if len(y_pred.shape) == 1 or (y_pred.shape[1] == 1 and y_true.shape[1] == 1):
            m = {
                f"{step_type}_{split}_{self.name}": self.calc_metric(
                    y_true.flatten(), y_pred.flatten()
                )
            }

        # Multi-binary classification-like  y_true and y_pred are shape=(batch, class)
        elif y_true.shape[1] != 1 and y_pred.shape[1] != 1:
            m = {
                f"{step_type}_{split}_{l}_{self.name}": self.calc_metric(yt, yp)
                for yt, yp, l in zip(y_true.T, y_pred.T, labels)
            }

        # Multi-class classification-like  y_true is shape=(batch, 1) or shape=(batch,) and y_pred is shape=(batch, class)
        elif (len(y_true.shape) == 1 or y_true.shape[1] == 1) and y_pred.shape[1] != 1:
            m = {
                f"{step_type}_{split}_{l}_{self.name}": self.calc_metric(
                    y_true.flatten() == i, yp
                )
                for i, (yp, l) in enumerate(
                    zip(y_pred.T, labels)
                )  # turn multi class into binary classification
            }

        return m


class CumulativeMetric(TrainingMetric):

    """Wraps a metric to apply to every class in output and calculate a cumulative value (like mean AUC)"""

    def __init__(
        self,
        training_metric: TrainingMetric,
        metric_func,
        metric_name="cumulative",
        optimum=None,
    ):
        optimum = optimum or training_metric.optimum
        metric_name = f"{metric_name}_{training_metric.name}"
        super().__init__(metric_func, metric_name, optimum)
        self.base_metric = training_metric

    def __call__(self, y_true, y_pred, labels=None, split=None, step_type=None):
        vals = list(self.base_metric(y_true, y_pred, labels, split, step_type).values())

        m = {f"{step_type}_{split}_{self.name}": self.func(vals)}
        return m


r2_metric = TrainingMetric(skl_metrics.r2_score, "r2", optimum="max")
roc_auc_metric = TrainingMetric(skl_metrics.roc_auc_score, "roc_auc", optimum="max")
accuracy_metric = TrainingMetric(skl_metrics.accuracy_score, "accuracy", optimum="max")
mae_metric = TrainingMetric(skl_metrics.mean_absolute_error, "mae", optimum="min")
pred_value_mean_metric = TrainingMetric(
    lambda y_true, y_pred: np.mean(y_pred), "pred_value_mean"
)
pred_value_std_metric = TrainingMetric(
    lambda y_true, y_pred: np.std(y_pred), "pred_value_std"
)


class TrainingModel(pl.LightningModule):
    def __init__(
        self,
        model,
        metrics: Iterable[TrainingMetric] = dict(),
        tracked_metric=None,
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__()
        self.epoch_preds = {"train": ([], []), "val": ([], [])}
        self.epoch_losses = {"train": [], "val": []}
        self.metrics = {}
        self.metric_funcs = {m.name: m for m in metrics}
        self.tracked_metric = f"epoch_val_{tracked_metric}"
        self.best_tracked_metric = None
        self.early_stop_epochs = early_stop_epochs
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.metrics["epochs_since_last_best"] = 0
        self.m = model
        self.training_steps = 0
        self.steps_since_checkpoint = 0
        self.labels = index_labels
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]
        if isinstance(save_predictions_path, str):
            save_predictions_path = Path(save_predictions_path)
        self.save_predictions_path = save_predictions_path
        self.lr = lr
        self.step_loss = (None, None)

        self.log_path = Path(wandb.run.dir) if wandb.run is not None else None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)

    def forward(self, x: dict):
        # if anything other than 'primary_input' and 'extra_inputs' is used,
        # this function must be overridden 
        if 'extra_inputs' in x:
            return self.m((x['primary_input'], x['extra_inputs']))
        else:
            return self.m(x['primary_input'])

    def step(self, batch, step_type='train'):
        batch = self.prepare_batch(batch)
        y_pred = self.forward(batch)

        if step_type != 'predict':
            if 'labels' not in batch:
                batch['labels'] = torch.empty(0)
            loss = self.loss_func(y_pred, batch['labels'])
            if torch.isnan(loss):
                raise ValueError(loss)

            self.log_step(step_type, batch['labels'], y_pred, loss)

            return loss
        else:
            return y_pred
    
    def prepare_batch(self, batch):
        return batch

    def training_step(self, batch, i):
        return self.step(batch, "train")

    def validation_step(self, batch, i):
        return self.step(batch, "val")

    def predict_step(self, batch, *args):
        y_pred = self.step(batch, "predict")
        return {"filename": batch["filename"], "prediction": y_pred.cpu().numpy()}

    def on_predict_epoch_end(self, results):

        for i, predict_results in enumerate(results):
            filename_df = pd.DataFrame(
                {
                    "filename": np.concatenate(
                        [batch["filename"] for batch in predict_results]
                    )
                }
            )

            if self.labels is not None:
                columns = [f"{class_name}_preds" for class_name in self.labels]
            else:
                columns = ["preds"]
            outputs_df = pd.DataFrame(
                np.concatenate(
                    [batch["prediction"] for batch in predict_results], axis=0
                ),
                columns=columns,
            )

            prediction_df = pd.concat([filename_df, outputs_df], axis=1)

            dataloader = self.trainer.predict_dataloaders[i]
            manifest = dataloader.dataset.manifest
            prediction_df = prediction_df.merge(manifest, on="filename", how="outer")
            if wandb.run is not None:
                prediction_df.to_csv(
                    Path(wandb.run.dir).parent
                    / "data"
                    / f"dataloader_{i}_predictions.csv",
                    index=False,
                )
            if self.save_predictions_path is not None:

                if ".csv" in self.save_predictions_path.name:
                    prediction_df.to_csv(
                        self.save_predictions_path.parent
                        / self.save_predictions_path.name.replace(".csv", f"_{i}_.csv"),
                        index=False,
                    )
                else:
                    prediction_df.to_csv(
                        self.save_predictions_path / f"dataloader_{i}_potassium_predictions.csv",
                        index=False,
                    )

            if wandb.run is None and self.save_predictions_path is None:
                print(
                    "WandB is not active and self.save_predictions_path is None. Predictions will be saved to the directory this script is being run in."
                )
                prediction_df.to_csv(f"dataloader_{i}_predictions.csv", index=False)

    def log_step(self, step_type, labels, output_tensor, loss):
        self.step_loss = (step_type, loss.detach().item())
        self.epoch_preds[step_type][0].append(labels.detach().cpu().numpy())
        self.epoch_preds[step_type][1].append(output_tensor.detach().cpu().numpy())
        self.epoch_losses[step_type].append(loss.detach().item())
        if step_type == "train":
            self.training_steps += 1
            self.steps_since_checkpoint += 1
            if (
                self.checkpoint_every_n_steps is not None
                and self.steps_since_checkpoint > self.checkpoint_every_n_steps
            ):
                self.steps_since_checkpoint = 0
                self.checkpoint_weights(f"step_{self.training_steps}")

    def checkpoint_weights(self, name=""):
        if wandb.run is not None:
            weights_path = Path(wandb.run.dir).parent / "weights"
            if not weights_path.is_dir():
                weights_path.mkdir()
            torch.save(self.state_dict(), weights_path / f"model_{name}.pt")
        else:
            print("Did not checkpoint model. wandb not initialized.")

    def validation_epoch_end(self, preds):

        # Save weights
        self.metrics["epoch"] = self.current_epoch
        if self.checkpoint_every_epoch:
            self.checkpoint_weights(f"epoch_{self.current_epoch}")

        # Calculate metrics
        for m_type in ["train", "val"]:

            y_true, y_pred = self.epoch_preds[m_type]
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

            self.metrics[f"epoch_{m_type}_loss"] = np.mean(self.epoch_losses[m_type])
            for m in self.metric_funcs.values():
                self.metrics.update(
                    m(
                        y_true,
                        y_pred,
                        labels=self.labels,
                        split=m_type,
                        step_type="epoch",
                    )
                )

            # Reset predictions
            self.epoch_losses[m_type] = []
            self.epoch_preds[m_type] = ([], [])

        # Check if new best epoch
        if self.metrics is not None and self.tracked_metric is not None:
            if self.tracked_metric == "epoch_val_loss":
                metric_optimization = "min"
            else:
                metric_optimization = self.metric_funcs[
                    self.tracked_metric.replace("epoch_val_", "")
                ].optimum
            if (
                self.metrics[self.tracked_metric] is not None
                and (
                    self.best_tracked_metric is None
                    or (
                        metric_optimization == "max"
                        and self.metrics[self.tracked_metric] > self.best_tracked_metric
                    )
                    or (
                        metric_optimization == "min"
                        and self.metrics[self.tracked_metric] < self.best_tracked_metric
                    )
                )
                and self.current_epoch > 0
            ):
                print(
                    f"New best epoch! {self.tracked_metric}={self.metrics[self.tracked_metric]}, epoch={self.current_epoch}"
                )
                self.checkpoint_weights(f"best_{self.tracked_metric}")
                self.metrics["epochs_since_last_best"] = 0
                self.best_tracked_metric = self.metrics[self.tracked_metric]
            else:
                self.metrics["epochs_since_last_best"] += 1
            if self.metrics["epochs_since_last_best"] >= self.early_stop_epochs:
                raise KeyboardInterrupt("Early stopping condition met")

        # Log to w&b
        if wandb.run is not None:
            wandb.log(self.metrics)


class RegressionModel(TrainingModel):
    def __init__(
        self,
        model,
        metrics=(r2_metric, mae_metric, pred_value_mean_metric, pred_value_std_metric),
        tracked_metric="mae",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__(
            model=model,
            metrics=metrics,
            tracked_metric=tracked_metric,
            early_stop_epochs=early_stop_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            index_labels=index_labels,
            save_predictions_path=save_predictions_path,
            lr=lr,
        )
        self.loss_func = nn.MSELoss()

    def prepare_batch(self, batch):
        if "labels" in batch and len(batch["labels"].shape) == 1:
            batch["labels"] = batch["labels"][:, None]
        return batch


class BinaryClassificationModel(TrainingModel):
    def __init__(
        self,
        model,
        metrics=(roc_auc_metric, CumulativeMetric(roc_auc_metric, np.nanmean, "mean")),
        tracked_metric="mean_roc_auc",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__(
            model=model,
            metrics=metrics,
            tracked_metric=tracked_metric,
            early_stop_epochs=early_stop_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            index_labels=index_labels,
            save_predictions_path=save_predictions_path,
            lr=lr,
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    def prepare_batch(self, batch):
        if "labels" in batch and len(batch["labels"].shape) == 1:
            batch["labels"] = batch["labels"][:, None]
        return batch


# Addresses bug caused by labels from a single column in a manifest being delivered as Bx1,
# but nn.CrossEntropyLoss wants a simple list of length B.
class SqueezeCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.cross_entropy(y_pred, y_true.squeeze(dim=-1))


class MultiClassificationModel(TrainingModel):
    def __init__(
        self,
        model,
        metrics=(roc_auc_metric, CumulativeMetric(roc_auc_metric, np.mean, "mean")),
        tracked_metric="mean_roc_auc",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        metrics = [*metrics]
        super().__init__(
            model=model,
            metrics=metrics,
            tracked_metric=tracked_metric,
            early_stop_epochs=early_stop_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            index_labels=index_labels,
            save_predictions_path=save_predictions_path,
            lr=lr,
        )
        self.loss_func = SqueezeCrossEntropyLoss()

    def prepare_batch(self, batch):
        if "labels" in batch:
            batch["labels"] = batch["labels"].long()
        batch["primary_input"] = batch["primary_input"].float()
        return batch


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"

    m = torchvision.models.video.r2plus1d_18()
    m.fc = nn.Linear(512, 1)
    training_model = RegressionModel(m)
    x = torch.randn((4, 3, 8, 112, 112))
    y = m(x)
    print(y.shape)


