"""Optional predictive models with preprocessing inside train/test pipelines."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans


def _preprocessor(X):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = list(X.select_dtypes(exclude="number").columns)
    transformers = []
    if numeric: transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical: transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical))
    if not transformers: raise ValueError("Select at least one usable feature column.")
    return ColumnTransformer(transformers)


def train_predictor(df: pd.DataFrame, target: str, features: list[str], task: str, test_size: float = .2) -> dict:
    if target not in df.columns: raise ValueError("Choose a target column present in the dataset.")
    if not features or target in features: raise ValueError("Select one or more features, excluding the target (prevents target leakage).")
    if any(f not in df.columns for f in features): raise ValueError("One or more selected features are not in the dataset.")
    data = df[features + [target]].copy().dropna(subset=[target])
    if len(data) < 12: raise ValueError("At least 12 labeled rows are required for a train/test evaluation.")
    X, y = data[features], data[target]
    if y.nunique() < 2: raise ValueError("The target needs at least two distinct values.")
    if task == "classification":
        counts = y.value_counts()
        stratify = y if counts.min() >= 2 and y.nunique() < len(y) * .5 else None
        if stratify is None and y.nunique() > 20: raise ValueError("Classification target has too many unique classes; choose regression or another target.")
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        try: Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        except ValueError as exc: raise ValueError(f"Could not create a reliable classification split: {exc}") from exc
        pipe = Pipeline([("prep", _preprocessor(X)), ("model", model)])
        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
        metrics = {"accuracy": float(accuracy_score(yte, pred)), "weighted_f1": float(f1_score(yte, pred, average="weighted", zero_division=0))}
    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(y): raise ValueError("Regression requires a numeric target column.")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        pipe = Pipeline([("prep", _preprocessor(X)), ("model", LinearRegression())])
        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
        metrics = {"mean_absolute_error": float(mean_absolute_error(yte, pred)), "r2": float(r2_score(yte, pred)) if len(yte) > 1 else float("nan")}
    else: raise ValueError("Choose classification or regression.")
    return {"model": pipe, "metrics": metrics, "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "features": features, "target": target, "task": task, "predictions": pd.DataFrame({"actual": yte.reset_index(drop=True), "predicted": pd.Series(pred).reset_index(drop=True)})}


def cluster_records(df: pd.DataFrame, features: list[str], clusters: int = 3) -> pd.DataFrame:
    if not features: raise ValueError("Select numeric columns to cluster.")
    X = df[features].select_dtypes(include="number")
    if X.shape[1] == 0: raise ValueError("K-means needs at least one numeric feature.")
    if not 2 <= clusters <= min(10, len(df)-1): raise ValueError("Choose a cluster count from 2 to min(10, row count − 1).")
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformed = prep.fit_transform(X)
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(transformed)
    result = df.copy(); result["cluster"] = labels
    return result
