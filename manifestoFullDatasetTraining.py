import os
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import csv
import json
import numpy as np

# ====================================================
# AUTOMATIC GPU DETECTION
# ====================================================
try:
    # Try importing RAPIDS libraries (GPU-accelerated replacements for scikit-learn and pandas)
    import cuml, cudf, cupy
    gpu_available = True
    # Read GPU device name for display purposes
    gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
except Exception:
    # Fall back to CPU mode if RAPIDS is not installed or GPU not accessible
    gpu_available = False
    gpu_name = None

USE_GPU = gpu_available
print("⚙️  Compute mode:", "🟢 GPU" if USE_GPU else "🔵 CPU")
if gpu_available:
    print(f"🧩 GPU detected: {gpu_name}")
else:
    print("💡 Running on CPU (scikit-learn).")

# ====================================================
# CONFIGURATION
# ====================================================
DATASET_PATH = "training_dataset_manifesto.csv"  # Path to labeled training dataset
MODEL_PATH = "models/manifesto_classifier.joblib"  # Output file for trained model
OUTPUT_DIR = "output/manifestoTraining"            # Folder for saving results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download Spanish stopwords the first time
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# ====================================================
# CONDITIONAL IMPORTS (GPU OR CPU)
# ====================================================
if USE_GPU:
    # RAPIDS GPU versions
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import LabelEncoder
else:
    # CPU versions (scikit-learn)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import shuffle

# ====================================================
# LOAD AND CLEAN THE DATASET
# ====================================================
df = pd.read_csv(DATASET_PATH, low_memory=False)

# Ensure correct types and remove empty/super-short texts
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()
df = df[df["text"].str.split().str.len() >= 4]  # Filter extremely short entries
df = df.dropna(subset=["text", "label"])
df = shuffle(df, random_state=42)

print(f"✅ Dataset loaded: {len(df)} samples, {df['label'].nunique()} categories.")

# ====================================================
# STRATIFIED SPLITTING FUNCTION
# ====================================================
def stratified_split(df, label_col="label", test_size=0.2, min_val=1):
    """
    Performs a stratified manual split: each label keeps the same proportion
    in training and validation sets. Guarantees a minimum number of samples.
    """
    train_parts, val_parts = [], []
    for label, group in df.groupby(label_col):
        n_val = max(min_val, int(len(group) * test_size))
        group = group.sample(frac=1, random_state=42)
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])
    return pd.concat(train_parts), pd.concat(val_parts)

# Create train/validation splits
train_df, val_df = stratified_split(df, "label", 0.2, 1)
print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

# ====================================================
# TF-IDF VECTORIZATION
# ====================================================
print("🧠 Vectorizing text (TF-IDF)...")

# Large TF-IDF with n-grams up to 3, stopwords, pruning rare/common words
tfidf = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1, 3),
    stop_words=spanish_stopwords,
    sublinear_tf=True,
    min_df=2,          # Remove rare words
    max_df=0.9,        # Remove overly common words
    norm='l2'
)

if USE_GPU:
    # cuDF (GPU DataFrame) version
    X_train = tfidf.fit_transform(cudf.Series(train_df["text"]))
    X_val = tfidf.transform(cudf.Series(val_df["text"]))
else:
    X_train = tfidf.fit_transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])

print(f"✅ Vectorization completed. Shape: {X_train.shape}")

# ====================================================
# TRAINING
# ====================================================
print("🚀 Training model...")

if USE_GPU:
    # GPU Logistic Regression (cuml)
    lr = LogisticRegression(max_iter=500, fit_intercept=True, class_weight="balanced")
    le = LabelEncoder()
    y_train = le.fit_transform(cudf.Series(train_df["label"]))
    y_val = le.transform(cudf.Series(val_df["label"]))
else:
    # CPU Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])

    # Small grid search to tune regularization strength (C)
    grid_params = {"C": [0.5, 1.0, 2.0], "penalty": ["l2"]}
    base_lr = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1
    )
    search = GridSearchCV(base_lr, grid_params, cv=2, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    lr = search.best_estimator_
    print(f"🏆 Best parameter C found: {lr.C}")

# Progress-bar animation (cosmetic)
for _ in tqdm(range(5), desc="⚙️ Initializing training"):
    pass

# GPU: must explicitly call lr.fit
if USE_GPU:
    lr.fit(X_train, y_train)

print("✅ Training completed.")

# ====================================================
# EVALUATION
# ====================================================
print("📊 Evaluating model...")

if USE_GPU:
    # GPU → labels are cuDF Series, convert to pandas later
    preds_enc = lr.predict(X_val)
    preds = le.inverse_transform(preds_enc).to_pandas()
else:
    preds_enc = lr.predict(X_val)
    preds = le.inverse_transform(preds_enc)

# Compute accuracy and classification report
acc = accuracy_score(val_df["label"], preds)
report = classification_report(val_df["label"], preds, zero_division=0, output_dict=True)

print(f"\n🎯 Overall accuracy: {acc:.3f}\n")
print("📈 Detailed report:")
print(classification_report(val_df["label"], preds, zero_division=0))

# ====================================================
# SAVE METRICS AND TOP LABEL INFORMATION
# ====================================================
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"]
    }, f, indent=4, ensure_ascii=False)
print(f"📊 Metrics saved to: {metrics_path}")

# Top 20 most common labels in the dataset
top_labels = df["label"].value_counts().head(20).index
top_report = {k: v for k, v in report.items() if k in top_labels}

# Save metrics for the most common labels
pd.DataFrame(top_report).T.to_csv(
    os.path.join(OUTPUT_DIR, "top_labels_metrics.csv"),
    quoting=csv.QUOTE_ALL,
    encoding="utf-8-sig"
)

# ====================================================
# CONFUSION MATRIX (safe conversions for GPU/CPU)
# ====================================================
print("📉 Generating confusion matrix...")

def to_numpy_safe(arr):
    """Safely convert GPU (cuDF/cuPy) or CPU arrays into pure NumPy."""
    try:
        if hasattr(arr, "to_numpy"):
            return arr.to_numpy()
        elif hasattr(arr, "get"):
            return arr.get()
        elif isinstance(arr, np.ndarray):
            return arr
        else:
            return np.array(arr)
    except Exception as e:
        print(f"⚠️ NumPy conversion failed: {e}")
        return np.array(arr)

y_true_np = to_numpy_safe(y_val)
y_pred_np = to_numpy_safe(preds_enc)

cm = confusion_matrix(y_true_np, y_pred_np)

# Plot simplified confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (simplified)")
plt.xlabel("Predicted")
plt.ylabel("True")

conf_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(conf_path, dpi=300)
plt.close()
print(f"🖼️ Confusion matrix saved to: {conf_path}")

# ====================================================
# SAVE TRAINED MODEL
# ====================================================
os.makedirs("models", exist_ok=True)
joblib.dump({"vectorizer": tfidf, "model": lr, "encoder": le}, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")

# ====================================================
# MANUAL TEST SAMPLES
# ====================================================
samples = [
    "Promoveremos la igualdad de oportunidades entre hombres y mujeres.",
    "Reduciremos los impuestos a las pequeñas empresas.",
    "Reforzaremos la defensa nacional ante nuevas amenazas globales.",
    "Fomentaremos la descentralización y la autonomía regional."
]

if USE_GPU:
    # GPU version: use cuDF
    sample_series = cudf.Series(samples)
    X_samples = tfidf.transform(sample_series)
    pred_enc = lr.predict(X_samples)
    pred_labels = le.inverse_transform(pred_enc).to_pandas()
else:
    X_samples = tfidf.transform(samples)
    pred_enc = lr.predict(X_samples)
    pred_labels = le.inverse_transform(pred_enc)

# Print predictions for quick verification
print("\n🔍 Test predictions:")
for t, p in zip(samples, pred_labels):
    print(f"🗣️ '{t}'\n→ 📘 {p}\n")

# ====================================================
# EXPORT VALIDATION RESULTS
# ====================================================
val_df["predicted"] = preds
out_csv = os.path.join(OUTPUT_DIR, "validation_predictions.csv")

val_df.to_csv(
    out_csv,
    index=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    encoding="utf-8-sig"
)

print(f"📑 Validation results saved to: {out_csv}")
