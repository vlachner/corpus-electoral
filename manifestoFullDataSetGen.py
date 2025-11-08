import os
import pandas as pd
from tqdm import tqdm
import csv

# ================================================
# CONFIGURACIÓN
# ================================================
BASE_DIR = "manifestoProjectDocs"
CODEBOOK_PATH = os.path.join(BASE_DIR, "codebook_categories_MPDS2020a.csv")
COUNTRIES_PATH = os.path.join(BASE_DIR, "MPDataset_MPDS2025a.csv")
OUTPUT_DATASET = "training_dataset_manifesto.csv"

# ================================================
# CARGAR CODEBOOK Y MAPA DE PARTIDOS/PAÍSES
# ================================================
df_codebook = pd.read_csv(CODEBOOK_PATH)
df_codebook["code"] = df_codebook["code"].astype(str).str.replace(".0", "", regex=False)

df_countries = pd.read_csv(COUNTRIES_PATH, low_memory=False)

# Limpieza básica
df_countries["country"] = df_countries["country"].astype(str).str.strip()
df_countries["party"] = df_countries["party"].astype(str).str.strip()
df_countries["countryname"] = df_countries["countryname"].astype(str).str.strip()
df_countries["partyname"] = df_countries["partyname"].astype(str).str.strip()

country_map = dict(zip(df_countries["country"], df_countries["countryname"]))
party_map = dict(zip(df_countries["party"], df_countries["partyname"]))

# ================================================
# RECOLECCIÓN GLOBAL DE MANIFIESTOS
# ================================================
rows = []

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".csv"):
            continue

        # Ignorar archivos de referencia
        if any(ex in file for ex in ["codebook", "MPDataset_MPDS2025a", "ListOfCountriesInfo"]):
            continue

        file_path = os.path.join(root, file)
        try:
            df = pd.read_csv(file_path)

            if "cmp_code" not in df.columns:
                continue

            # Convertir códigos a texto
            df["cmp_code"] = df["cmp_code"].astype(str).str.replace(".0", "", regex=False)

            # Merge con el codebook
            merged = df.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

            # -----------------------------------------------
            # 🔍 Identificar país y partido desde el filename
            # -----------------------------------------------
            file_id = file.split("_")[0]

            # Buscar coincidencia de país (2 o 3 dígitos)
            country_candidates = sorted(df_countries["country"].unique(), key=len, reverse=True)
            country_code = None
            for candidate in country_candidates:
                if file_id.startswith(candidate):
                    country_code = candidate
                    break

            country_name = country_map.get(country_code, "Unknown Country") if country_code else "Unknown Country"

            # ⚠️ El party_code es el código completo (country + party)
            party_code = file_id
            party_row = df_countries.loc[df_countries["party"] == party_code, "partyname"]
            party_name = party_row.values[0] if not party_row.empty else "Unknown"
            year = file.split("_")[-1].replace(".csv", "")

            # -----------------------------------------------
            # 📜 Buscar la columna de texto
            # -----------------------------------------------
            text_col = None
            for c in ["text", "sentence", "content", "quasi_sentence"]:
                if c in merged.columns:
                    text_col = c
                    break

            if text_col is None:
                continue

            # Filtrar solo texto y etiqueta
            subset = merged[[text_col, "title"]].copy()

            # Reemplazar NaN o vacíos por "No category"
            subset["title"] = subset["title"].fillna("No category")

            # Limpiar texto
            subset[text_col] = (
                subset[text_col]
                .astype(str)
                .replace({r"\s+": " "}, regex=True)
                .str.strip()
            )

            subset["country"] = country_name
            subset["party"] = party_name
            subset["year"] = year
            subset["source_file"] = os.path.splitext(file)[0]

            # Debug opcional
            total = len(subset)
            no_cat = (subset["title"] == "No category").sum()
            print(f"📄 {file}: {total} filas ({no_cat} sin categoría)")

            rows.append(subset)

        except Exception as e:
            print(f"⚠️ Error leyendo {file_path}: {e}")

# ================================================
# CONCATENAR TODO Y GUARDAR
# ================================================
if rows:
    df_all = pd.concat(rows, ignore_index=True)
    df_all.rename(columns={text_col: "text", "title": "label"}, inplace=True)

    df_all.to_csv(
        OUTPUT_DATASET,
        index=False,
        quoting=csv.QUOTE_ALL,  # asegura comillas en todo
        quotechar='"',
        encoding="utf-8-sig"
    )

    print(f"\n✅ Dataset generado con {len(df_all)} ejemplos -> {OUTPUT_DATASET}")
else:
    print("⚠️ No se generaron filas. Revisa los nombres de columnas en tus CSVs.")
