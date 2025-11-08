import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
BASE_DIR = "manifestoProjectDocs"
INPUT_CODEBOOK = os.path.join(BASE_DIR, "codebook_categories_MPDS2020a.csv")
INPUT_COUNTRIES = os.path.join(BASE_DIR, "MPDataset_MPDS2025a.csv")
OUTPUT_DIR = "output/manifestoResults"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CARGAR CODEBOOK Y MAPA DE PAÍSES
# ============================================================
df_codebook = pd.read_csv(INPUT_CODEBOOK)
df_codebook["code"] = df_codebook["code"].astype(str).str.replace(".0", "", regex=False)

df_countries = pd.read_csv(INPUT_COUNTRIES)

# ⚠️ NO eliminar las columnas de partido
# Solo aseguramos que existan y limpiamos duplicados relevantes
df_countries = df_countries[["country", "countryname", "party", "partyname"]].drop_duplicates()

# Crear mapa simple de países
country_map = dict(zip(df_countries["country"].astype(str), df_countries["countryname"]))

# 🔍 Mostrar estructura leída
print("\n===== 🧩 CODEBOOK (primeras filas) =====")
print(df_codebook.head())

print("\n===== 🌍 COUNTRIES (primeras filas) =====")
print(df_countries.head())

print("\n===== 📚 Columnas detectadas =====")
print("CODEBOOK:", list(df_codebook.columns))
print("COUNTRIES:", list(df_countries.columns))

# ============================================================
# CÓDIGOS IDEOLÓGICOS (RILE)
# ============================================================
right_codes = [104, 201, 203, 305, 401, 402, 407, 414, 505, 507,
               601, 603, 605, 606, 608, 809]
left_codes  = [103, 105, 106, 107, 403, 404, 406, 412, 413,
               504, 506, 701, 202]

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
        if "cmp_code" not in df_manifesto.columns:
            print(f"⚠️ {csv_path} no contiene columna 'cmp_code'. Saltando...")
            return None

        df_manifesto["cmp_code"] = df_manifesto["cmp_code"].astype(str).str.replace(".0", "", regex=False)
        merged = df_manifesto.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

        # Crear carpeta de salida
        filename = os.path.basename(csv_path)
        year = filename.split("_")[-1].replace(".csv", "")
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

        if not category_counts.empty:
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

        print(f"✅ Procesando archivo: {os.path.basename(csv_path)} ({country_name}/{party_name} {year}): RILE = {rile:.2f}")
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
# RECORRER TODAS LAS SUBCARPETAS RECURSIVAMENTE
# ============================================================
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".csv"):
            continue

        # Archivos que deben ser ignorados explícitamente
        if any(excluded in file for excluded in [
            "codebook", "ListOfCountriesInfo", "MPDataset_MPDS2025a", "countries", "readme"
        ]):
            continue

        csv_path = os.path.join(root, file)
        # Extraer parte numérica antes del "_"
        file_id = file.split("_")[0]

        # Determinar dinámicamente el código de país según los que existen en el dataset
        country_candidates = sorted(df_countries["country"].astype(str).unique(), key=len, reverse=True)
        country_code = None

        for candidate in country_candidates:
            if file_id.startswith(candidate):
                country_code = candidate
                break

        # Si no se encontró coincidencia, marcar como desconocido
        if country_code is None:
            country_name = "Unknown Country"
        else:
            country_name = country_map.get(country_code, "Unknown Country")

        # Asignar partido a partir del resto del ID
        party_code = file_id
        party_name = df_countries.loc[
            df_countries["party"].astype(str) == party_code,
            "partyname"
        ]
        party_name = party_name.values[0] if not party_name.empty else "Unknown"


        res = process_manifesto(csv_path, country_name, party_name)
        if res:
            results.append(res)
        
        if country_code is None:
            print(f"⚠️ No se detectó país válido para {file_id}")

# ============================================================
# GUARDAR TABLAS GLOBALES + ANÁLISIS DE PARETO
# ============================================================
if results:
    df_results = pd.DataFrame(results).sort_values(["country", "party", "year"])
    df_results["ideology_class"] = df_results["rile"].apply(classify_rile)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "rile_summary.csv"), index=False)
    print("\n📈 Tabla global de RILE guardada con clasificación ideológica.")

# ---------- SUBTEMAS GLOBALES ----------
if global_subtopics:
    df_subtopics = pd.concat(global_subtopics, ignore_index=True)
    total_subtopics = (
        df_subtopics.groupby("title")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_subtopics["% Acumulado"] = (total_subtopics["Frecuencia"].cumsum() / total_subtopics["Frecuencia"].sum()) * 100
    total_subtopics.to_csv(os.path.join(OUTPUT_DIR, "subtopics_global.csv"), index=False)
    print("📘 CSV global de subtemas guardado: subtopics_global.csv")

    # Pareto
    plt.figure(figsize=(10, 6))
    plt.bar(total_subtopics["title"].head(20), total_subtopics["Frecuencia"].head(20), color="#007acc")
    plt.plot(total_subtopics["title"].head(20), total_subtopics["% Acumulado"].head(20), color="orange", marker="o")
    plt.title("Distribución tipo Pareto de Subtemas MARPOR (Top 20)")
    plt.xlabel("Subtema MARPOR")
    plt.ylabel("Frecuencia / % Acumulado")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_subtopics_global.png"))
    plt.close()
    print("📊 Gráfico Pareto de subtemas guardado: pareto_subtopics_global.png")

# ---------- TEMAS GLOBALES ----------
if global_topics:
    df_topics = pd.concat(global_topics, ignore_index=True)
    total_topics = (
        df_topics.groupby("Tema")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_topics["% Acumulado"] = (total_topics["Frecuencia"].cumsum() / total_topics["Frecuencia"].sum()) * 100
    total_topics.to_csv(os.path.join(OUTPUT_DIR, "topics_global.csv"), index=False)
    print("📗 CSV global de temas guardado: topics_global.csv")

    # Pareto
    plt.figure(figsize=(10, 6))
    plt.bar(total_topics["Tema"].head(20), total_topics["Frecuencia"].head(20), color="#f39c12")
    plt.plot(total_topics["Tema"].head(20), total_topics["% Acumulado"].head(20), color="orange", marker="o")
    plt.title("Distribución tipo Pareto de Temas MARPOR (Top 20)")
    plt.xlabel("Tema MARPOR")
    plt.ylabel("Frecuencia / % Acumulado")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_topics_global.png"))
    plt.close()
    print("📊 Gráfico Pareto de temas guardado: pareto_topics_global.png")
