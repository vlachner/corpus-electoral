import os
import pandas as pd
from tqdm import tqdm
import csv

# ================================================
# CONFIGURATION
# ================================================
BASE_DIR = "manifestoProjectDocs"  
# Root directory where all manifesto CSV files are stored.

CODEBOOK_PATH = os.path.join(BASE_DIR, "codebook_categories_MPDS2020a.csv")
# MARPOR codebook containing mapping from cmp_code → title/category.

COUNTRIES_PATH = os.path.join(BASE_DIR, "MPDataset_MPDS2025a.csv")
# Dataset containing country codes, country names, party codes, and party names.

OUTPUT_DATASET = "training_dataset_manifesto.csv"
# Final output dataset used for ML training (text + label).

# ================================================
# LOAD CODEBOOK AND COUNTRY/PARTY MAPPINGS
# ================================================
df_codebook = pd.read_csv(CODEBOOK_PATH)
# Normalize codebook codes (remove trailing ".0" from float-to-string conversion).
df_codebook["code"] = df_codebook["code"].astype(str).str.replace(".0", "", regex=False)

df_countries = pd.read_csv(COUNTRIES_PATH, low_memory=False)

# Basic cleanup to ensure consistent formatting of identifiers.
df_countries["country"] = df_countries["country"].astype(str).str.strip()
df_countries["party"] = df_countries["party"].astype(str).str.strip()
df_countries["countryname"] = df_countries["countryname"].astype(str).str.strip()
df_countries["partyname"] = df_countries["partyname"].astype(str).str.strip()

# Maps for retrieving country and party names from their codes.
country_map = dict(zip(df_countries["country"], df_countries["countryname"]))
party_map = dict(zip(df_countries["party"], df_countries["partyname"]))

# ================================================
# GLOBAL COLLECTION OF MANIFESTO LINES
# ================================================
rows = []  # Will store extracted (text, label, metadata) rows from all manifestos.

# Walk recursively through manifestoProjectDocs directory.
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".csv"):
            continue  # Skip non-CSV files.

        # Ignore reference metadata CSV files.
        if any(ex in file for ex in ["codebook", "MPDataset_MPDS2025a", "ListOfCountriesInfo"]):
            continue

        file_path = os.path.join(root, file)

        try:
            df = pd.read_csv(file_path)

            # If there is no code column, the file is not a manifesto.
            if "cmp_code" not in df.columns:
                continue

            # Normalize cmp_code format.
            df["cmp_code"] = df["cmp_code"].astype(str).str.replace(".0", "", regex=False)

            # Merge manifesto data with codebook to attach category titles.
            merged = df.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

            # -----------------------------------------------
            # 🔍 IDENTIFY COUNTRY AND PARTY FROM FILENAME
            # -----------------------------------------------
            file_id = file.split("_")[0]  
            # Example filename: "35210_PartyName_2019.csv"
            # The prefix "35210" contains both the country code and the party code.

            # Try matching the longest possible country code first.
            country_candidates = sorted(df_countries["country"].unique(), key=len, reverse=True)
            country_code = None

            for candidate in country_candidates:
                if file_id.startswith(candidate):
                    country_code = candidate
                    break

            # Resolve country name (fallback to Unknown)
            country_name = country_map.get(country_code, "Unknown Country") if country_code else "Unknown Country"

            # Party code is the full prefix (country + party)
            party_code = file_id
            party_row = df_countries.loc[df_countries["party"] == party_code, "partyname"]
            # If party not found in country dataset → label as Unknown
            party_name = party_row.values[0] if not party_row.empty else "Unknown"

            # Extract year from filename (after the last "_")
            year = file.split("_")[-1].replace(".csv", "")

            # -----------------------------------------------
            # 📜 DETECT TEXT COLUMN
            # -----------------------------------------------
            text_col = None
            for c in ["text", "sentence", "content", "quasi_sentence"]:
                if c in merged.columns:
                    text_col = c
                    break

            # If no text column exists, skip this file
            if text_col is None:
                continue

            # Extract text + category (title)
            subset = merged[[text_col, "title"]].copy()

            # Replace missing titles with a fallback label
            subset["title"] = subset["title"].fillna("No category")

            # Clean text: collapse whitespace and remove leading/trailing spaces
            subset[text_col] = (
                subset[text_col]
                .astype(str)
                .replace({r"\s+": " "}, regex=True)
                .str.strip()
            )

            # Attach metadata fields
            subset["country"] = country_name
            subset["party"] = party_name
            subset["year"] = year
            subset["source_file"] = os.path.splitext(file)[0]  # file without extension

            # Optional debug info
            total = len(subset)
            no_cat = (subset["title"] == "No category").sum()
            print(f"📄 {file}: {total} rows ({no_cat} without category)")

            rows.append(subset)

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

# ================================================
# CONCATENATE EVERYTHING AND SAVE FINAL DATASET
# ================================================
if rows:
    df_all = pd.concat(rows, ignore_index=True)

    # Rename columns so the ML pipeline always receives consistent names
    df_all.rename(columns={text_col: "text", "title": "label"}, inplace=True)

    # Save dataset in CSV with full quoting (protects commas/newlines)
    df_all.to_csv(
        OUTPUT_DATASET,
        index=False,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        encoding="utf-8-sig"
    )

    print(f"\n✅ Dataset generated with {len(df_all)} examples -> {OUTPUT_DATASET}")
else:
    print("⚠️ No rows were generated. Check column names inside your manifesto CSV files.")
