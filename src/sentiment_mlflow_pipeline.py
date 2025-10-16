# =============================================================
# Experiment: Full MLflow Pipeline with Text Classification
# =============================================================

import pandas as pd
import numpy as np
import warnings
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
print("✅ Libraries imported successfully.")

# =============================================================
# Step 1: Load and Prepare Dataset
# =============================================================
try:
    df = pd.read_excel("cleaned_twitter_dataset_with_sentiment.xlsx")
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

    # Basic cleaning
    df = df.dropna(subset=["text", "sentiment"])

    # Encode sentiment labels as numbers
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df["sentiment"] = df["sentiment"].map(label_map)

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

except FileNotFoundError:
    print("❌ Dataset not found. Please check the file name.")
    exit()




# =============================================================
# Step 2: Preprocessor & Model Configurations
# =============================================================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

models_and_params = {
    "LinearSVC": {
        "model": LinearSVC(random_state=42, max_iter=10000),
        "params": {
            "classifier__C": [0.1, 1, 10]
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=10000, random_state=42),
        "params": {
            "classifier__C": [0.1, 1, 10]
        },
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        ),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2]
        },
    },
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)
mlflow.set_experiment("TwitterSentiment_Experiment")

# =============================================================
# Step 3: Train and Log Models
# =============================================================
print("\n--- 🚀 Training Models with MLflow ---")

for name, config in models_and_params.items():
    # Baseline
    with mlflow.start_run(run_name=f"Baseline_{name}"):
        mlflow.set_tag("model_type", "Baseline")

        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", config["model"])
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("test_accuracy", acc)
        print(f"  - {name} baseline accuracy: {acc:.4f}")

    # Tuned
    if config["params"]:
        with mlflow.start_run(run_name=f"Tuned_{name}"):
            mlflow.set_tag("model_type", "Tuned")

            pipeline = Pipeline([
                ("tfidf", vectorizer),
                ("classifier", config["model"])
            ])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=config["params"],
                n_iter=4,
                cv=cv,
                scoring="accuracy",
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("model_name", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_accuracy", search.best_score_)
            mlflow.sklearn.log_model(search.best_estimator_, "model")

            print(f"  - {name} tuned accuracy: {acc:.4f}")

print("--- ✅ Training complete ---")

# =============================================================
# Step 4: Compare Results
# =============================================================
print("\n--- 📊 Comparing Model Results ---")
runs_df = mlflow.search_runs(order_by=["metrics.best_cv_accuracy DESC"])
comparison = []
for name in models_and_params.keys():
    base = runs_df[(runs_df["tags.model_type"] == "Baseline") & (runs_df["params.model_name"] == name)]
    tuned = runs_df[(runs_df["tags.model_type"] == "Tuned") & (runs_df["params.model_name"] == name)]
    if not base.empty and not tuned.empty:
        base_acc = base.iloc[0].get("metrics.test_accuracy", np.nan)
        tuned_acc = tuned.iloc[0].get("metrics.best_cv_accuracy", np.nan)
        comparison.append({
            "Model": name,
            "Baseline": base_acc,
            "Tuned": tuned_acc,
            "Improvement": tuned_acc - base_acc
        })

comp_df = pd.DataFrame(comparison)
print(comp_df)

# =============================================================
# Step 5: Visualize
# =============================================================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Tuned", data=comp_df, color="lightgreen")
plt.title("Model Accuracy Comparison (Tuned Models)")
plt.ylabel("Accuracy")
plt.show()

# =============================================================
# Step 6: Save Best Model
# =============================================================
if not comp_df.empty:
    best_model_name = comp_df.loc[comp_df["Tuned"].idxmax()]["Model"]
    best_run = runs_df[
        (runs_df["tags.model_type"] == "Tuned") &
        (runs_df["params.model_name"] == best_model_name)
    ].iloc[0]

    best_run_id = best_run["run_id"]
    best_model_pipeline = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model_pipeline, f)

    print(f"🏆 Best model saved as 'best_model.pkl' ({best_model_name})")

