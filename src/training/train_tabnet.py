import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import json
import uuid
from datetime import datetime
from itertools import product
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from src.utils.io import load_pickle
from pytorch_tabnet.metrics import Metric
from pandas.plotting import parallel_coordinates
import optuna
import optuna.visualization as vis

class CostScore(Metric):
    def __init__(self, alpha=5.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self._name = "custom_cost"
        self._maximize = False

    def __call__(self, y_true, y_score):
        if y_score.ndim == 2:
            y_pred = np.argmax(y_score, axis=1)
        else:
            y_pred = (y_score > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        return self.alpha * fnr + self.beta * fpr

def save_fold_models(fold_models, params, fold_metrics, path_prefix="models/tabnet_cv"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    base_path = f"{path_prefix}_{timestamp}_{uid}"
    os.makedirs(base_path, exist_ok=True)

    model_paths = []

    for i, clf in enumerate(fold_models):
        model_path = f"{base_path}/fold_{i+1}"
        clf.save_model(model_path)
        model_paths.append(model_path)

    metadata = {
        "timestamp": timestamp,
        "uid": uid,
        "base_path": base_path,
        "model_paths": model_paths,
        "hyperparameters": params,
        "validation_metrics": fold_metrics
    }

    with open(f"{base_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Alle Fold-Modelle gespeichert unter: {base_path}")
    print(f"[✓] Metadaten gespeichert unter: {base_path}/metadata.json")

def train_tabnet(X_train, y_train, X_val, y_val, params, fold_idx=None):
    clf = TabNetClassifier(
        **params,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
    )

    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)],
        eval_name=["train", "val"],
        eval_metric=["logloss"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    return clf

def plot_learning_curves(history, fold_idx, out_dir="reports/learning_curves"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_logloss"]) + 1)

    plt.figure(figsize=(6, 5))

    plt.plot(epochs, history["train_logloss"], label="Train Logloss")
    plt.plot(epochs, history["val_logloss"], label="Val Logloss")
    plt.xlabel("Epoch")
    plt.ylabel("Logloss")
    plt.title(f"Logloss Fold {fold_idx+1}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fold_{fold_idx+1}.png")
    plt.close()

def plot_final_metric_matrix(df, metrics, highlight_metric="avg_cost", out_path="reports/grid_search/final_metric_matrix.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = df.copy()
    df["config_label"] = df.apply(
        lambda r: f"{r['n_d']}-{r['n_a']}-{r['mask_type']}-{r['lambda_sparse']}", axis=1
    )

    x = list(range(len(df)))
    labels = df["config_label"].tolist()

    best_idx = df[highlight_metric].idxmin()
    best_x = x[best_idx]

    plt.figure(figsize=(len(df) * 0.5 + 3, 6))

    for metric in metrics:
        plt.scatter(x, df[metric], label=metric, s=60)

    for xi in x:
        plt.axvline(x=xi, color="gray", linestyle="--", alpha=0.2)

    # Highlight der besten Konfiguration
    plt.axvspan(best_x - 0.5, best_x + 0.5, color="red", alpha=0.1)

    for metric in metrics:
        plt.scatter(best_x, df.loc[best_idx, metric], 
                    s=150, edgecolors='black', linewidths=1.5, 
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][metrics.index(metric)],
                    zorder=5)
        
    plt.annotate("⬇", (best_x, plt.ylim()[1] * 0.99), 
             ha="center", va="top", fontsize=16, color="red")

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Konfiguration (n_d - n_a - mask_type - lambda_sparse)")
    plt.ylabel("Wert")
    plt.title("Metriken über alle Konfigurationen")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.,
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[✓] Grid-Sreach Metrikplot gespeichert unter: {out_path}")

def run_training(n_trials=20):
    print("[✓] Lade Trainingsdaten...")
    trainval_df = load_pickle("data/processed/train_val_pool.pkl")
    X_full = trainval_df.drop(columns=["attack_detected"])
    y_full = trainval_df["attack_detected"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trial_metrics_list = []

    def objective(trial):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 32, step=8),
            "n_a": trial.suggest_int("n_a", 8, 32, step=8),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            "lambda_sparse": trial.suggest_loguniform("lambda_sparse", 1e-5, 1e-3),
            "n_steps": trial.suggest_int("n_steps", 3, 5),
            "gamma": 1.5
        }

        fold_costs = []
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            clf = train_tabnet(X_train, y_train, X_val, y_val, params)
            plot_learning_curves(clf.history, fold_idx)

            y_val_pred = clf.predict(X_val.values)
            y_val_proba = clf.predict_proba(X_val.values)[:, 1]
            val_cost = CostScore()(y_val.values, y_val_proba)

            fold_costs.append(val_cost)

            fold_metrics.append({
                "cost": val_cost,
                "logloss": clf.history["val_logloss"][-1],
                "f1": f1_score(y_val, y_val_pred),
                "auc": roc_auc_score(y_val, y_val_proba),
                "precision": precision_score(y_val, y_val_pred),
                "recall": recall_score(y_val, y_val_pred),
                "accuracy": accuracy_score(y_val, y_val_pred)
            })

        # Logging für spätere Analyse
        trial_metrics_list.append({
            **params,
            "avg_cost": np.mean([m["cost"] for m in fold_metrics]),
            "f1": np.mean([m["f1"] for m in fold_metrics]),
            "precision": np.mean([m["precision"] for m in fold_metrics]),
            "recall": np.mean([m["recall"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "auc": np.mean([m["auc"] for m in fold_metrics]),
            "logloss": np.mean([m["logloss"] for m in fold_metrics])
        })

        return np.mean(fold_costs)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    os.makedirs("reports/optuna", exist_ok=True)

    vis.plot_optimization_history(study).write_html("reports/optuna/history.html")
    vis.plot_parallel_coordinate(study).write_html("reports/optuna/parallel_coordinates.html")
    vis.plot_param_importances(study).write_html("reports/optuna/importance.html")
    vis.plot_slice(study).write_html("reports/optuna/slice_plot.html")
    vis.plot_contour(study).write_html("reports/optuna/contour_plot.html")

    print("[✓] Optuna-Visualisierungen gespeichert unter: reports/optuna/")

    print("\n[✓] Bayesian Optimization abgeschlossen.")
    print(f"Beste Konfiguration: {study.best_params}")
    print(f"Bestwert (CostScore): {study.best_value:.4f}")

    # Bestes Modell final trainieren
    best_params = study.best_params
    best_params["gamma"] = 1.5

    fold_metrics = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        clf = train_tabnet(X_train, y_train, X_val, y_val, best_params)
        plot_learning_curves(clf.history, fold_idx)

        y_val_pred = clf.predict(X_val.values)
        y_val_proba = clf.predict_proba(X_val.values)[:, 1]
        val_cost = CostScore()(y_val.values, y_val_proba)

        fold_metrics.append({
            "cost": val_cost,
            "logloss": clf.history["val_logloss"][-1],
            "f1": f1_score(y_val, y_val_pred),
            "auc": roc_auc_score(y_val, y_val_proba),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "accuracy": accuracy_score(y_val, y_val_pred)
        })
        fold_models.append(clf)

    # Speichern + Visualisieren
    df_results = pd.DataFrame(trial_metrics_list)
    df_results.to_csv("reports/cv_summary/bayesopt_results.csv", index=False)

    metrics_to_plot = ["avg_cost", "f1", "precision", "recall", "accuracy", "auc"]
    plot_final_metric_matrix(
        df_results,
        metrics=metrics_to_plot,
        highlight_metric="avg_cost",
        out_path="reports/grid_search/final_metric_matrix.png"
    )

    save_fold_models(
        fold_models=fold_models,
        params=best_params,
        fold_metrics=fold_metrics,
        path_prefix="models/tabnet_bayesopt"
    )

    print("[✓] Finalmodelle gespeichert.")
