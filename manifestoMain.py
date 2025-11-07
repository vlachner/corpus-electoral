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
# FUNCIÓN DE PROCESAMIENTO
# ============================================================
def process_manifesto(csv_path, country_name, party_name):
    """Procesa un manifiesto individual, guarda gráficos, RILE y temas/subtemas."""
    try:
        # ---------- Cargar CSV ----------
        df_manifesto = pd.read_csv(csv_path)
        df_manifesto["cmp_code"] = df_manifesto["cmp_code"].astype(str).str.replace(".0", "", regex=False)
        merged = df_manifesto.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

        # ---------- Crear carpeta de salida ----------
        year = os.path.basename(csv_path).split("_")[-1].replace(".csv", "")
        party_output = os.path.join(OUTPUT_DIR, country_name, party_name)
        os.makedirs(party_output, exist_ok=True)

        # ====================================================
        # 1️⃣ TOP CATEGORÍAS (SUBTEMAS)
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

        # ====================================================
        # 2️⃣ DISTRIBUCIÓN POR TEMAS (MACROTEMAS)
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

            # Guardar CSV con temas
            domain_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_temas.csv"), index=False)

        # ====================================================
        # 3️⃣ CSV DE SUBTEMAS
        # ====================================================
        category_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_subtemas.csv"), index=False)

        # ====================================================
        # 4️⃣ CÁLCULO DEL RILE INDEX
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
        return {
            "country": country_name,
            "party": party_name,
            "year": year,
            "rile": rile,
            "num_subtemas": len(category_counts),
            "num_temas": len(domain_counts) if domain_column else 0
        }

    except Exception as e:
        print(f"❌ Error procesando {csv_path}: {e}")
        return None


# ============================================================
# RECORRER TODOS LOS ARCHIVOS EN LAS CARPETAS
# ============================================================
results = []
topic_summary = []

for base_path in BASE_PATHS:
    country_name = os.path.basename(os.path.dirname(base_path))  # "CostaRica" o "Uruguay"
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
# GUARDAR TABLAS RESUMEN
# ============================================================
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results.sort_values(["country", "party", "year"], inplace=True)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "rile_summary.csv"), index=False)
    print("\n📈 Tabla global de RILE guardada en output/manifestoResults/rile_summary.csv")
else:
    print("⚠️ No se generaron resultados válidos.")
