# Political Text Processing Pipeline – Dataset, Analysis & Classifier

This repository implements a **pipeline** for political text processing, mainly for data extraction, since data analysis is done using an exploratory language training. Also it includes PDF sentence extraction, OCR paragraph cleaning, Manifesto Project dataset construction, supervised classification, and MARPOR-based ideological analysis (including RILE).

It covers:

### ✔ Automated extraction of sentences from PDFs  
### ✔ Cleaning and segmentation of OCR-generated TXT paragraphs  
### ✔ Construction of a unified training dataset (MARPOR categories)  
### ✔ TF-IDF + Logistic Regression classifier (CPU/GPU compatible) -> only as an exploratory language processing
### ✔ MARPOR analytics: subtopics, domains, frequency distributions  
### ✔ RILE Index computation + ideological orientation classification  
### ✔ Global Pareto plots and CSV summaries  

All outputs are exported into structured folders for reproducibility.

---

# Project Structure

```
project/
│
├── manifestoFullDataSetGen.py
├── manifestoFullDatasetTraining.py
├── manifestoMain.py
├── main.py
├── crCorpusDataset2026.py
│
├── docs/                           # Raw PDF corpus
├── docs-PDFBot/                    # OCR-processed TXT files
├── manifestoProjectDocs/           # Official MARPOR datasets
│   ├── codebook_categories_MPDS2020a.csv
│   ├── MPDataset_MPDS2025a.csv
│   └── <country/party/year CSVs>
│
├── output/ -> not part of the repo, it is automatically generated
│   ├── manifestoResults/           # MARPOR analysis output
│   ├── manifestoTraining/          # Classifier results, metrics
│   └── exported datasets
│
└── README.md
```

---

# 🔧 Dependencies

Install core dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib nltk tqdm
```

Optional **GPU acceleration (RAPIDS)**:

```bash
conda install -c rapidsai -c nvidia -c conda-forge cuml cudf cupy
```

---

# Full Processing Pipeline

## **1. Extract sentences from PDF files**
**Script:** `main.py`  -> this script is partially used for data extraction of PDFs
✔ Scans all PDFs  
✔ Extracts sentences + page numbers  
✔ Adds metadata (author, document type, year)  
✔ Saves timestamped CSV

---

## **2. Clean OCR paragraphs from TXT**
**Script:** `crCorpusDataset2026.py`  
**Description:** This is used mainly to construct the final dataset, it takes all the generated outputs from PDFBot, and other paragraph extraction tools, the extrated text resides in txt format in folder docs-PDFBot. This script is to generate Costa Rica corpus for the government plans from 2026 only.
✔ Detects true paragraph boundaries  
✔ Reconstructs broken lines  
✔ Removes noise and excess whitespace  
✔ Generates `corpus_parrafos.csv`

---

## **3. Build the complete MARPOR dataset**
**Script:** `manifestoFullDataSetGen.py`  
**Description:** The datasets from manifesto were downloaded from their repository, only for hispanic countries, and this script unifies all of them in a single file:
✔ Reads all Manifesto Project CSV files  
✔ Joins quasi-sentences with MARPOR categories  
✔ Attaches country, party, year metadata  
✔ Creates `training_dataset_manifesto.csv`

---

## **4. Train the supervised classifier**
**Script:** `manifestoFullDatasetTraining.py`
**Description:** This is just an exploratory model training, to understand some metrics:
### Features:
- TF-IDF (1–3 grams), stopword filtering  
- Logistic Regression (CPU or GPU)  
- GridSearch tuning (CPU)  
- Full metrics: accuracy, classification report, confusion matrix  
- Top-20 labels analysis  
- Saves model + vectorizer

### Outputs:
```
models/manifesto_classifier.joblib
output/manifestoTraining/
```

---

## **5. Perform MARPOR analysis + RILE computation**
**Script:** `manifestoMain.py`

### Generates:
- Top subtopics by party/year  
- Domain/macrotopic distributions  
- Global subtopic and topic frequency tables  
- Pareto charts  
- Per-party RILE Index  
- Ideology classification:
  - Far Left  
  - Left  
  - Center-Left  
  - Center-Right  
  - Right  
  - Far Right

### Saved to:
```
output/manifestoResults/
```

---

# Script-by-Script Overview

---

## **main.py** — PDF Sentence Extractor  
Scans the `docs/` directory, extracts sentences and metadata, and exports a timestamped dataset.

Output example:
```
output/political_sentences_dataset_<timestamp>.csv
```

---

## **crCorpusDataset2026.py** — OCR Paragraph Cleaner  
Processes TXT files from OCR, reconstructs paragraphs, and outputs a clean corpus.

Output:
```
corpus_parrafos.csv
```

---

## **manifestoFullDataSetGen.py** — Build Manifesto Training Dataset  
Combines MARPOR codebook, MPDataset, and per-party CSV files into a unified dataset.

Output:
```
training_dataset_manifesto.csv
```

---

## **manifestoFullDatasetTraining.py** — TF-IDF Classifier Training  
Implements the classifier pipeline (CPU/GPU) and saves metrics, plots, and the trained model.

Outputs:
```
models/manifesto_classifier.joblib
output/manifestoTraining/
```

---

## **manifestoMain.py** — MARPOR Analysis + RILE  
Performs full aggregation, visualization, and ideological scoring.

Outputs include:
- Subtopic charts  
- Domain frequency plots  
- RILE bar charts  
- Global Pareto charts  
- CSV tables for all analytics  

Saved in:
```
output/manifestoResults/
```

---

# How to Run the Full Pipeline

```bash
python main.py
python crCorpusDataset2026.py
python manifestoFullDataSetGen.py
python manifestoFullDatasetTraining.py
python manifestoMain.py
```

---

# Example Workflow

```bash
# 1 — Extract sentences from PDFs
python main.py

# 2 — Clean OCR paragraphs
python crCorpusDataset2026.py

# 3 — Build complete Manifesto dataset
python manifestoFullDataSetGen.py

# 4 — Train the classifier
python manifestoFullDatasetTraining.py

# 5 — Perform MARPOR analytics
python manifestoMain.py
```

---

# Summary

This repository provides a robust and reproducible workflow for:

- **Political text extraction and normalization**  
- **Dataset engineering using Manifesto Project standards**  
- **Supervised classification of MARPOR categories**  
- **Ideological analysis via RILE Index**  
- **Global corpus statistics and visualization**

The modular scripts allow you to run each stage independently or perform the entire end-to-end process.

## References

### PDF24 Tools  
PDF24. *(n.d.).* **PDF24 Tools – PDF converter and utilities.** Retrieved from  
<https://tools.pdf24.org/>

### Manifesto Project (Main Website)  
Volkens, A., Krause, W., Lehmann, P., Matthieß, T., Merz, N., Regel, S., & Weßels, B. *(2024).* **The Manifesto Project.** Wissenschaftszentrum Berlin für Sozialforschung (WZB).  
Available at: <https://manifesto-project.wzb.eu/>

### ManifestoBERTa / XLM-RoBERTa Model (Hugging Face)  
Manifesto Project. *(2024).* **ManifestoBERTa / XLM-RoBERTa: 56 Policy Topics Sentence Model (2024-1-1)** \[Model\]. Hugging Face.  
<https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1>

### PDFBoT (GitHub)  
Zhang, C. *(2021).* **PDFBoT: PDF text and paragraph extraction toolkit** \[Software\]. GitHub repository.  
<https://github.com/ZhangChengX/PDFBoT>
