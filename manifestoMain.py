import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# GENERAL CONFIGURATION
# ============================================================
BASE_DIR = "manifestoProjectDocs"  
# Root folder containing all manifesto CSV files and metadata.

INPUT_CODEBOOK = os.path.join(BASE_DIR, "codebook_categories_MPDS2020a.csv")
# Codebook file that maps MARPOR codes to categories and descriptions.

INPUT_COUNTRIES = os.path.join(BASE_DIR, "MPDataset_MPDS2025a.csv")
# Dataset with country and party metadata (country codes, party names, etc.).

OUTPUT_DIR = "output/manifestoResults"
# Folder where all generated graphs and tables will be saved.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD CODEBOOK AND COUNTRY MAP
# ============================================================
df_codebook = pd.read_csv(INPUT_CODEBOOK)

# Normalize code column (remove trailing ".0" from floats stored as strings).
df_codebook["code"] = df_codebook["code"].astype(str).str.replace(".0", "", regex=False)

df_countries = pd.read_csv(INPUT_COUNTRIES)

# ⚠️ We do NOT remove party columns — keep party metadata intact.
# Keep only needed columns and remove duplicates.
df_countries = df_countries[["country", "countryname", "party", "partyname"]].drop_duplicates()

# Create mapping from country code → country name.
country_map = dict(zip(df_countries["country"].astype(str), df_countries["countryname"]))

# Show structures for debugging.
print("\n===== 🧩 CODEBOOK (first rows) =====")
print(df_codebook.head())

print("\n===== 🌍 COUNTRIES (first rows) =====")
print(df_countries.head())

print("\n===== 📚 Detected columns =====")
print("CODEBOOK:", list(df_codebook.columns))
print("COUNTRIES:", list(df_countries.columns))

# ============================================================
# IDEOLOGICAL CODE GROUPS (RILE)
# ============================================================
right_codes = [104, 201, 203, 305, 401, 402, 407, 414, 505, 507,
               601, 603, 605, 606, 608, 809]
# MARPOR codes typically associated with "Right" ideological positions.

left_codes  = [103, 105, 106, 107, 403, 404, 406, 412, 413,
               504, 506, 701, 202]
# MARPOR codes typically associated with "Left" ideological positions.

# ============================================================
# GLOBAL ACCUMULATORS
# ============================================================
results = []            # Store RILE results for each manifesto.
global_subtopics = []  # Store all sub-category counts globally.
global_topics = []     # Store all macro-topic counts globally.

# ============================================================
# MANIFESTO PROCESSING FUNCTION
# ============================================================
def process_manifesto(csv_path, country_name, party_name):
    """Processes a single manifesto: merges with codebook, computes stats,
    generates plots, and returns RILE score."""

    try:
        df_manifesto = pd.read_csv(csv_path)

        # Skip files without MARPOR code column.
        if "cmp_code" not in df_manifesto.columns:
            print(f"⚠️ {csv_path} does not contain 'cmp_code'. Skipping...")
            return None

        # Normalize MARPOR codes.
        df_manifesto["cmp_code"] = df_manifesto["cmp_code"].astype(str).str.replace(".0", "", regex=False)

        # Merge with codebook to obtain categories and domain information.
        merged = df_manifesto.merge(df_codebook, how="left", left_on="cmp_code", right_on="code")

        # Prepare output folder for this party/country/year
        filename = os.path.basename(csv_path)
        year = filename.split("_")[-1].replace(".csv", "")
        party_output = os.path.join(OUTPUT_DIR, country_name, party_name)
        os.makedirs(party_output, exist_ok=True)

        # ====================================================
        # 1️⃣ SUBTOPICS (MARPOR categories)
        # ====================================================
        category_counts = (
            merged.groupby(["cmp_code", "title"])
            .size()
            .reset_index(name="Frecuencia")
            .sort_values("Frecuencia", ascending=False)
        )

        # Plot top 15 subtopics.
        if not category_counts.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(category_counts["title"].head(15),
                     category_counts["Frecuencia"].head(15),
                     color="#007acc")
            plt.title(f"{country_name} – {party_name} {year}\nTop 15 MARPOR subtopics")
            plt.xlabel("Number of quasi-sentences")
            plt.ylabel("MARPOR subtopic (title)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(party_output, f"{party_name}_{year}_top15_subtemas.png"))
            plt.close()

            # Save CSV for subtopics.
            category_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_subtemas.csv"), index=False)

            # Add to global accumulator.
            category_counts["country"] = country_name
            category_counts["party"] = party_name
            category_counts["year"] = year
            global_subtopics.append(category_counts)

        # ====================================================
        # 2️⃣ MACROTOPICS (Domain / Main Class)
        # ====================================================
        domain_column = None
        for candidate in ["domain_name", "domain", "main_class"]:
            # Auto-detect which column is available depending on MPDS version.
            if candidate in merged.columns:
                domain_column = candidate
                break

        # Plot domain distribution if available.
        if domain_column:
            domain_counts = merged[domain_column].value_counts().reset_index()
            domain_counts.columns = ["Tema", "Frecuencia"]

            plt.figure(figsize=(8, 5))
            plt.barh(domain_counts["Tema"], domain_counts["Frecuencia"], color="#2ca02c")
            plt.title(f"{country_name} – {party_name} {year}\nMARPOR domain distribution")
            plt.xlabel("Frequency")
            plt.ylabel("Main topic")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(party_output, f"{party_name}_{year}_temas.png"))
            plt.close()

            # Save domain CSV
            domain_counts.to_csv(os.path.join(party_output, f"{party_name}_{year}_temas.csv"), index=False)

            # Add to global accumulator
            domain_counts["country"] = country_name
            domain_counts["party"] = party_name
            domain_counts["year"] = year
            global_topics.append(domain_counts)

        # ====================================================
        # 3️⃣ RILE INDEX CALCULATION
        # ====================================================
        # Convert codes to numeric so we can match them against RILE lists.
        df_manifesto["cmp_code"] = pd.to_numeric(df_manifesto["cmp_code"], errors="coerce")

        right_count = df_manifesto[df_manifesto["cmp_code"].isin(right_codes)].shape[0]
        left_count  = df_manifesto[df_manifesto["cmp_code"].isin(left_codes)].shape[0]

        # Avoid division by zero.
        rile = (right_count - left_count) / (right_count + left_count) if (right_count + left_count) > 0 else 0

        # Plot RILE bar chart.
        plt.figure(figsize=(6, 4))
        plt.bar(["Left", "Right"], [left_count, right_count], color=["red", "blue"])
        plt.title(f"{country_name} – {party_name} {year}\nIdeological balance (RILE = {rile:.2f})")
        plt.ylabel("Frequency of topics")
        plt.tight_layout()
        plt.savefig(os.path.join(party_output, f"{party_name}_{year}_rile_index.png"))
        plt.close()

        print(f"✅ Processing file: {os.path.basename(csv_path)} ({country_name}/{party_name} {year}): RILE = {rile:.2f}")
        return {"country": country_name, "party": party_name, "year": year, "rile": rile}

    except Exception as e:
        print(f"❌ Error processing {csv_path}: {e}")
        return None

# ============================================================
# IDEOLOGY CLASSIFICATION BASED ON RILE SCORE
# ============================================================
def classify_rile(r):
    """Categorize RILE score into ideological ranges."""
    if r <= -0.60:
        return "Far Left"
    elif r <= -0.20:
        return "Left"
    elif r <= 0.10:
        return "Center-Left"
    elif r <= 0.40:
        return "Center-Right"
    elif r <= 0.70:
        return "Right"
    else:
        return "Far Right"

# ============================================================
# RECURSIVELY WALK THROUGH ALL SUBFOLDERS
# ============================================================
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".csv"):
            continue

        # Ignore metadata or non-manifesto files.
        if any(excluded in file for excluded in [
            "codebook", "ListOfCountriesInfo", "MPDataset_MPDS2025a", "countries", "readme"
        ]):
            continue

        csv_path = os.path.join(root, file)
        
        # Extract manifesto ID (the prefix before "_").
        file_id = file.split("_")[0]

        # Try to detect the country code dynamically.
        country_candidates = sorted(df_countries["country"].astype(str).unique(),
                                    key=len, reverse=True)
        country_code = None

        for candidate in country_candidates:
            if file_id.startswith(candidate):
                country_code = candidate
                break

        # Assign country name (fallback to Unknown).
        country_name = country_map.get(country_code, "Unknown Country") if country_code else "Unknown Country"

        # Match party code with party name.
        party_code = file_id
        party_name = df_countries.loc[
            df_countries["party"].astype(str) == party_code,
            "partyname"
        ]
        party_name = party_name.values[0] if not party_name.empty else "Unknown"

        # Process manifesto file and store results.
        res = process_manifesto(csv_path, country_name, party_name)
        if res:
            results.append(res)

        if country_code is None:
            print(f"⚠️ No valid country detected for {file_id}")

# ============================================================
# SAVE GLOBAL TABLES + PARETO ANALYSIS
# ============================================================
if results:
    df_results = pd.DataFrame(results).sort_values(["country", "party", "year"])
    df_results["ideology_class"] = df_results["rile"].apply(classify_rile)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "rile_summary.csv"), index=False)
    print("\n📈 Global RILE table saved with ideological classification.")

# ---------- GLOBAL SUBTOPICS ----------
if global_subtopics:
    df_subtopics = pd.concat(global_subtopics, ignore_index=True)

    # Aggregate subtopics globally.
    total_subtopics = (
        df_subtopics.groupby("title")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_subtopics["% Accumulated"] = (
        total_subtopics["Frecuencia"].cumsum() / total_subtopics["Frecuencia"].sum()
    ) * 100

    total_subtopics.to_csv(os.path.join(OUTPUT_DIR, "subtopics_global.csv"), index=False)
    print("📘 Global subtopics CSV saved: subtopics_global.csv")

    # Pareto chart for subtopics.
    plt.figure(figsize=(10, 6))
    plt.bar(total_subtopics["title"].head(20),
            total_subtopics["Frecuencia"].head(20),
            color="#007acc")
    plt.plot(total_subtopics["title"].head(20),
             total_subtopics["% Accumulated"].head(20),
             color="orange", marker="o")
    plt.title("Pareto Distribution of MARPOR Subtopics (Top 20)")
    plt.xlabel("MARPOR Subtopic")
    plt.ylabel("Frequency / % Accumulated")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_subtopics_global.png"))
    plt.close()
    print("📊 Global subtopics Pareto chart saved: pareto_subtopics_global.png")

# ---------- GLOBAL TOPICS ----------
if global_topics:
    df_topics = pd.concat(global_topics, ignore_index=True)

    # Aggregate topics globally.
    total_topics = (
        df_topics.groupby("Tema")["Frecuencia"]
        .sum()
        .reset_index()
        .sort_values("Frecuencia", ascending=False)
    )
    total_topics["% Accumulated"] = (
        total_topics["Frecuencia"].cumsum() / total_topics["Frecuencia"].sum()
    ) * 100

    total_topics.to_csv(os.path.join(OUTPUT_DIR, "topics_global.csv"), index=False)
    print("📗 Global topics CSV saved: topics_global.csv")

    # Pareto chart for topics.
    plt.figure(figsize=(10, 6))
    plt.bar(total_topics["Tema"].head(20),
            total_topics["Frecuencia"].head(20),
            color="#f39c12")
    plt.plot(total_topics["Tema"].head(20),
             total_topics["% Accumulated"].head(20),
             color="orange", marker="o")
    plt.title("Pareto Distribution of MARPOR Topics (Top 20)")
    plt.xlabel("MARPOR Topic")
    plt.ylabel("Frequency / % Accumulated")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_topics_global.png"))
    plt.close()
    print("📊 Global topics Pareto chart saved: pareto_topics_global.png")
