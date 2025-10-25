import pandas as pd
from tqdm import tqdm
import src.utils as utils
import src.extractSentencesMethods as extractSentencesMethods
from datetime import datetime
import os

PDF_ROOT = "docs"
OUTPUT_FOLDER = "output"

# Crear carpeta si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def build_dataset(root_dir):
    rows = []
    # collect all PDF paths first
    pdf_files = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, fname))

    print("Found", len(pdf_files), "PDF files")

    for fpath in tqdm(pdf_files, desc="Processing PDFs"):
        author, doc_type, year = utils.parse_path_for_metadata(fpath)
        print(author, doc_type, year)
        sentences = extractSentencesMethods.extract_sentences_from_pdf(fpath)
        for sent, page_number in sentences:
            rows.append({
                "author": author,
                "document_type": doc_type,
                "year": year,
                "sentence": sent,
                "pdf_path": fpath,
                "page_number": page_number
            })
    return pd.DataFrame(rows)

def build_dataset_single_file(file_path):
    """
    Procesa un PDF específico y construye un DataFrame con sus oraciones relevantes.
    """
    rows = []

    print("Processing file:", file_path)

    # Extraer metadata si quieres
    author, doc_type, year = utils.parse_path_for_metadata(file_path)
    print(author, doc_type, year)

    # Extraer oraciones
    sentences = extractSentencesMethods.extract_sentences_from_pdf(file_path)

    for sent, page_number in sentences:
        rows.append({
            "author": author,
            "document_type": doc_type,
            "year": year,
            "sentence": sent,
            "pdf_path": file_path,
            "page_number": page_number
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    
    df = build_dataset(PDF_ROOT)

    # Format timestamp (safe for filenames)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename with timestamp
    filename = f"political_sentences_dataset_{ts}.csv"
    # Path completo
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    df.to_csv(filepath, index=False)

    print("Saved:", filename)