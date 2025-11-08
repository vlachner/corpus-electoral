import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
BASE_PATHS = [
    "manifestoProjectDocs/CostaRica/csvsAnnotatedText",
    "manifestoProjectDocs/Uruguay/csvAnnotatedText"
]
INPUT_CODEBOOK = "manifestoProjectDocs/codebook_categories_MPDS2020a.csv"
OUTPUT_DIR = "output/manifestoResults"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CARGAR CODEBOOK
# ============================================================
df_codebook = pd.read_csv(INPUT_CODEBOOK)
df_codebook["code"] = df_codebook["code"].astype(str).str.replace(".0", "", regex=False)

# ============================================================
# CÓDIGOS IDEOLÓGICOS (RILE)
# ============================================================
right_codes = [104, 201, 203, 305, 401, 402, 407, 414, 505, 507, 601, 603, 605, 606, 608, 809]
left_codes  = [103, 105, 106, 107, 403, 404, 406, 412, 413, 504, 506, 701, 202]

# ============================================================
# ACUMULADORES GLOBALES
# ============================================================
results = []
global_subtopics = []
global_topics = []

# ============================================================
# FUNCIÓN DE PROCESAMIENTO
# ============================================================
def process_manifesto(csv_path, country_name, party_name):
    """Procesa un manifiesto individual, guarda gráficos y actualiza totales globales."""
    try:
        df_manifesto = pd.read_csv(csv_path)
        df_manifesto["cmp_code"] = df_manifesto["cmp_code"].astype(str).str.replace(".0", "", regex=False)
        merged = df_manifesto.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

        # Crear carpeta de salida
        year = os.path.basename(csv_path).split("_")[-1].replace(".csv", "")
        party_output = os.path.join(OUTPUT_DIR, country_name, party_name)
        os.makedirs(party_output, exist_ok=True)

        # ====================================================
        # 1️⃣ SUBTEMAS (CATEGORÍAS MARPOR)
        # ====================================================
        category_counts = (
            merged.groupby(["cmp_code", "title"])
            .size()
            .reset_index(name="Frecuencia")
            .sort_values("Frecuencia", ascending=False)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(category_counts["title"].head(15),
                 category_counts["Frecuencia"].head(15),
                 color="#007acc")
        plt.title(f"{country_name} – {party_name} {year}\nTop 15 subtemas MARPOR")
        plt.xlabel("Número de quasi-sentencias")
        plt.ylabel("Subtema MARPOR (title)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(party_output, f"{party_name}_{year}_top15_subtemas.png"))
        plt.close()

        category_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_subtemas.csv"), index=False)

        # Agregar al global
        category_counts["country"] = country_name
        category_counts["party"] = party_name
        category_counts["year"] = year
        global_subtopics.append(category_counts)

        # ====================================================
        # 2️⃣ TEMAS (MACROTEMAS)
        # ====================================================
        domain_column = None
        for candidate in ["domain_name", "domain", "main_class"]:
            if candidate in merged.columns:
                domain_column = candidate
                break

        if domain_column:
            domain_counts = merged[domain_column].value_counts().reset_index()
            domain_counts.columns = ["Tema", "Frecuencia"]

            plt.figure(figsize=(8, 5))
            plt.barh(domain_counts["Tema"], domain_counts["Frecuencia"], color="#2ca02c")
            plt.title(f"{country_name} – {party_name} {year}\nDistribución de temas MARPOR")
            plt.xlabel("Frecuencia")
            plt.ylabel("Tema principal")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(party_output, f"{party_name}_{year}_temas.png"))
            plt.close()

            domain_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_temas.csv"), index=False)

            # Agregar al global
            domain_counts["country"] = country_name
            domain_counts["party"] = party_name
            domain_counts["year"] = year
            global_topics.append(domain_counts)

        # ====================================================
        # 3️⃣ CÁLCULO DEL RILE INDEX
        # ====================================================
        df_manifesto["cmp_code"] = pd.to_numeric(df_manifesto["cmp_code"], errors="coerce")
        right_count = df_manifesto[df_manifesto["cmp_code"].isin(right_codes)].shape[0]
        left_count  = df_manifesto[df_manifesto["cmp_code"].isin(left_codes)].shape[0]
        rile = (right_count - left_count) / (right_count + left_count) if (right_count + left_count) > 0 else 0

        plt.figure(figsize=(6, 4))
        plt.bar(["Izquierda", "Derecha"], [left_count, right_count], color=["red", "blue"])
        plt.title(f"{country_name} – {party_name} {year}\nBalance ideológico (RILE = {rile:.2f})")
        plt.ylabel("Frecuencia de temas")
        plt.tight_layout()
        plt.savefig(os.path.join(party_output, f"{party_name}_{year}_rile_index.png"))
        plt.close()

        print(f"✅ {country_name}/{party_name} {year}: RILE = {rile:.2f}")

        return {"country": country_name, "party": party_name, "year": year, "rile": rile}

    except Exception as e:
        print(f"❌ Error procesando {csv_path}: {e}")
        return None

# ============================================================
# CLASIFICACIÓN IDEOLÓGICA POR RANGO DE RILE
# ============================================================
def classify_rile(r):
    if r <= -0.60:
        return "Extrema Izquierda"
    elif r <= -0.20:
        return "Izquierda"
    elif r <= 0.10:
        return "Centro-Izquierda"
    elif r <= 0.40:
        return "Centro-Derecha"
    elif r <= 0.70:
        return "Derecha"
    else:
        return "Extrema Derecha"
# ============================================================
# RECORRER TODAS LAS CARPETAS
# ============================================================
for base_path in BASE_PATHS:
    country_name = os.path.basename(os.path.dirname(base_path))
    for party in sorted(os.listdir(base_path)):
        party_path = os.path.join(base_path, party)
        if os.path.isdir(party_path):
            csv_files = [f for f in os.listdir(party_path) if f.endswith(".csv")]
            for csv_file in csv_files:
                csv_path = os.path.join(party_path, csv_file)
                res = process_manifesto(csv_path, country_name, party)
                if res:
                    results.append(res)

# ============================================================
# GUARDAR TABLAS GLOBALES
# ============================================================
if results:
    df_results = pd.DataFrame(results).sort_values(["country", "party", "year"])
    df_results["ideology_class"] = df_results["rile"].apply(classify_rile)
    # Guardar CSV actualizado
    df_results.to_csv(os.path.join(OUTPUT_DIR, "rile_summary.csv"), index=False)
    print("\n📈 Tabla de RILE global guardada con clasificación ideológica.")

# ---------- Consolidar subtemas globales ----------
if global_subtopics:
    df_subtopics = pd.concat(global_subtopics, ignore_index=True)
    total_subtopics = (
        df_subtopics.groupby("title")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_subtopics.to_csv(os.path.join(OUTPUT_DIR, "subtopics_global.csv"), index=False)
    print("📘 CSV global de subtemas guardado: subtopics_global.csv")

# ---------- Consolidar temas globales ----------
if global_topics:
    df_topics = pd.concat(global_topics, ignore_index=True)
    total_topics = (
        df_topics.groupby("Tema")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_topics.to_csv(os.path.join(OUTPUT_DIR, "topics_global.csv"), index=False)
    print("📗 CSV global de temas guardado: topics_global.csv")

    # Gráfico global de temas
    plt.figure(figsize=(10, 6))
    plt.barh(total_topics["Tema"].head(15), total_topics["Frecuencia"].head(15), color="#f39c12")
    plt.title("Top 15 temas más frecuentes (Global Costa Rica + Uruguay)")
    plt.xlabel("Frecuencia total")
    plt.ylabel("Tema MARPOR")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "topics_global_top15.png"))
    plt.close()
    print("📊 Gráfico global de temas guardado: topics_global_top15.png")
