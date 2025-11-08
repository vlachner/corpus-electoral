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
# DETECCIÓN AUTOMÁTICA DE GPU
# ====================================================
try:
    import cuml, cudf, cupy
    gpu_available = True
    gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
except Exception:
    gpu_available = False
    gpu_name = None

USE_GPU = gpu_available
print("⚙️  Modo de cómputo:", "🟢 GPU" if USE_GPU else "🔵 CPU")
if gpu_available:
    print(f"🧩 GPU detectada: {gpu_name}")
else:
    print("💡 Ejecutando en CPU (scikit-learn).")

# ====================================================
# CONFIGURACIÓN
# ====================================================
DATASET_PATH = "training_dataset_manifesto.csv"
MODEL_PATH = "models/manifesto_classifier.joblib"
OUTPUT_DIR = "output/manifestoTraining"
os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# ====================================================
# IMPORTAR LIBRERÍAS SEGÚN ENTORNO
# ====================================================
if USE_GPU:
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import LabelEncoder
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import shuffle

# ====================================================
# CARGAR Y LIMPIAR EL DATASET
# ====================================================
df = pd.read_csv(DATASET_PATH, low_memory=False)
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()
df = df[df["text"].str.split().str.len() >= 4]  # eliminar textos muy cortos
df = df.dropna(subset=["text", "label"])
df = shuffle(df, random_state=42)

print(f"✅ Dataset cargado: {len(df)} ejemplos, {df['label'].nunique()} categorías.")

# ====================================================
# DIVISIÓN ESTRATIFICADA
# ====================================================
def stratified_split(df, label_col="label", test_size=0.2, min_val=1):
    train_parts, val_parts = [], []
    for label, group in df.groupby(label_col):
        n_val = max(min_val, int(len(group) * test_size))
        group = group.sample(frac=1, random_state=42)
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])
    return pd.concat(train_parts), pd.concat(val_parts)

train_df, val_df = stratified_split(df, "label", 0.2, 1)
print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

# ====================================================
# VECTORIZACIÓN TF-IDF
# ====================================================
print("🧠 Vectorizando texto (TF-IDF)...")
tfidf = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1, 3),
    stop_words=spanish_stopwords,
    sublinear_tf=True,
    min_df=2,          # elimina términos raros
    max_df=0.9,        # elimina términos demasiado comunes
    norm='l2'
)

if USE_GPU:
    X_train = tfidf.fit_transform(cudf.Series(train_df["text"]))
    X_val = tfidf.transform(cudf.Series(val_df["text"]))
else:
    X_train = tfidf.fit_transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])

print(f"✅ Vectorización completa. Dimensiones: {X_train.shape}")

# ====================================================
# ENTRENAMIENTO
# ====================================================
print("🚀 Entrenando modelo...")
if USE_GPU:
    lr = LogisticRegression(max_iter=500, fit_intercept=True, class_weight="balanced")
    le = LabelEncoder()
    y_train = le.fit_transform(cudf.Series(train_df["label"]))
    y_val = le.transform(cudf.Series(val_df["label"]))
else:
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])

    # 🔍 Pequeño grid search para optimizar regularización
    grid_params = {"C": [0.5, 1.0, 2.0], "penalty": ["l2"]}
    base_lr = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", n_jobs=-1)
    search = GridSearchCV(base_lr, grid_params, cv=2, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    lr = search.best_estimator_
    print(f"🏆 Mejor parámetro C encontrado: {lr.C}")

for _ in tqdm(range(5), desc="⚙️ Inicializando entrenamiento"):
    pass

if USE_GPU:
    lr.fit(X_train, y_train)

print("✅ Entrenamiento completado.")

# ====================================================
# EVALUACIÓN
# ====================================================
print("📊 Evaluando modelo...")
if USE_GPU:
    preds_enc = lr.predict(X_val)
    preds = le.inverse_transform(preds_enc).to_pandas()
else:
    preds_enc = lr.predict(X_val)
    preds = le.inverse_transform(preds_enc)

acc = accuracy_score(val_df["label"], preds)
report = classification_report(val_df["label"], preds, zero_division=0, output_dict=True)

print(f"\n🎯 Accuracy general: {acc:.3f}\n")
print("📈 Reporte detallado:")
print(classification_report(val_df["label"], preds, zero_division=0))

# ====================================================
# GUARDAR MÉTRICAS Y TOP LABELS
# ====================================================
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"]
    }, f, indent=4, ensure_ascii=False)
print(f"📊 Métricas guardadas en: {metrics_path}")

# top 20 etiquetas más frecuentes
top_labels = df["label"].value_counts().head(20).index
top_report = {k: v for k, v in report.items() if k in top_labels}
pd.DataFrame(top_report).T.to_csv(
    os.path.join(OUTPUT_DIR, "top_labels_metrics.csv"),
    quoting=csv.QUOTE_ALL,
    encoding="utf-8-sig"
)

# ====================================================
# MATRIZ DE CONFUSIÓN (conversión segura)
# ====================================================
print("📉 Generando matriz de confusión...")

def to_numpy_safe(arr):
    """Convierte cuDF/cuPy/NumPy a NumPy puro de forma segura."""
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
        print(f"⚠️ Conversión a NumPy falló: {e}")
        return np.array(arr)

y_true_np = to_numpy_safe(y_val)
y_pred_np = to_numpy_safe(preds_enc)

cm = confusion_matrix(y_true_np, y_pred_np)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Matriz de confusión (simplificada)")
plt.xlabel("Predicho")
plt.ylabel("Real")

conf_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(conf_path, dpi=300)
plt.close()
print(f"🖼️ Matriz de confusión guardada en: {conf_path}")

# ====================================================
# GUARDADO DEL MODELO
# ====================================================
os.makedirs("models", exist_ok=True)
joblib.dump({"vectorizer": tfidf, "model": lr, "encoder": le}, MODEL_PATH)
print(f"💾 Modelo guardado en: {MODEL_PATH}")

# ====================================================
# PRUEBAS MANUALES
# ====================================================
samples = [
    "Promoveremos la igualdad de oportunidades entre hombres y mujeres.",
    "Reduciremos los impuestos a las pequeñas empresas.",
    "Reforzaremos la defensa nacional ante nuevas amenazas globales.",
    "Fomentaremos la descentralización y la autonomía regional."
]

if USE_GPU:
    sample_series = cudf.Series(samples)
    X_samples = tfidf.transform(sample_series)
    pred_enc = lr.predict(X_samples)
    pred_labels = le.inverse_transform(pred_enc).to_pandas()
else:
    X_samples = tfidf.transform(samples)
    pred_enc = lr.predict(X_samples)
    pred_labels = le.inverse_transform(pred_enc)

print("\n🔍 Predicciones de prueba:")
for t, p in zip(samples, pred_labels):
    print(f"🗣️ '{t}'\n→ 📘 {p}\n")

# ====================================================
# EXPORTAR RESULTADOS
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
print(f"📑 Resultados guardados en: {out_csv}")
