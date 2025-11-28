import pandas as pd
from tqdm import tqdm
import src.utils as utils
import src.extractSentencesMethods as extractSentencesMethods
from datetime import datetime
import os

PDF_ROOT = "docs"        # Root folder containing all PDFs to be scanned.
OUTPUT_FOLDER = "output" # Where extracted CSV datasets will be stored.

# Create output folder if it doesn’t already exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def build_dataset(root_dir):
    """
    Recursively scans a directory for PDF files, extracts sentences from each PDF,
    attaches metadata (author, document type, year), and returns a DataFrame.

    Parameters:
        root_dir (str): Root folder containing PDF files.

    Returns:
        pd.DataFrame: A table containing one row per extracted sentence.
    """
    rows = []
    pdf_files = []

    # ----------------------------------------------------------------------
    # Collect all PDF filenames recursively before processing (pre-scan step)
    # ----------------------------------------------------------------------
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, fname))

    print("Found", len(pdf_files), "PDF files")

    # ----------------------------------------------------------------------
    # Process each PDF and extract sentences
    # ----------------------------------------------------------------------
    for fpath in tqdm(pdf_files, desc="Processing PDFs"):

        # Extract metadata based on file path conventions defined in utils.parse_path_for_metadata
        author, doc_type, year = utils.parse_path_for_metadata(fpath)
        print(author, doc_type, year)

        # Extract list of (sentence, page_number) pairs
        sentences = extractSentencesMethods.extract_sentences_from_pdf(fpath)

        # Store one row per extracted sentence
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
    Processes a single PDF and extracts all relevant sentences into a DataFrame.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        pd.DataFrame: A table containing extracted sentences and metadata.
    """
    rows = []

    print("Processing file:", file_path)

    # Extract metadata from file path (author, document type, year)
    author, doc_type, year = utils.parse_path_for_metadata(file_path)
    print(author, doc_type, year)

    # Extract sentences from the PDF
    sentences = extractSentencesMethods.extract_sentences_from_pdf(file_path)

    # Build row list
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
    
    # ------------------------------------------------------------
    # Build dataset from ALL PDFs under PDF_ROOT
    # ------------------------------------------------------------
    df = build_dataset(PDF_ROOT)

    # Create timestamp (safe for filenames, no spaces or colons)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct output filename
    filename = f"political_sentences_dataset_{ts}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    # Save extracted dataset
    df.to_csv(filepath, index=False)

    print("Saved:", filename)
