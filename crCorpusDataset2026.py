import os
import re
import pandas as pd

# ==========================
# CONFIGURATION
# ==========================
DIRECTORIO = "docs-PDFBot/2026"   # <--- CHANGE THIS FOLDER PATH IF NEEDED
OUTPUT_CSV = "corpus_parrafos.csv"

# Regex: detect double newlines, allowing invisible spaces/tabs between them.
# This is used to split the text into paragraphs.
DOUBLE_NL_PATTERN = re.compile(r'\n[ \t]*\n')

def procesar_archivo(ruta):
    """
    Reads a TXT file and returns a list of clean paragraphs.

    Steps:
    1. Load the file as UTF-8.
    2. Normalize line breaks.
    3. Split by paragraphs (double newline).
    4. Inside each paragraph, join broken lines into a single line.
    5. Remove extra spaces.
    """

    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    # Normalize all newline formats → use "\n"
    contenido = contenido.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs using the double newline pattern
    parrafos = DOUBLE_NL_PATTERN.split(contenido)

    parrafos_limpios = []
    for p in parrafos:

        # Replace single newlines inside a paragraph with spaces
        # (?<!\n) = the newline is not part of a double newline
        # (?!\n) = does not precede another newline
        p_limpio = re.sub(r'(?<!\n)\n[ \t]*(?!\n)', ' ', p)

        # Collapse multiple spaces/tabs → single space
        p_limpio = re.sub(r'[ \t]+', ' ', p_limpio).strip()

        # Ignore empty paragraphs
        if p_limpio:
            parrafos_limpios.append(p_limpio)

    return parrafos_limpios


def main():
    """
    Processes all TXT files inside the directory and produces a CSV with:

        doc_id   → numerical ID per file
        title    → filename without extension
        text     → one paragraph per row

    The doc_id increments ONCE per file, not per paragraph.
    """
    filas = []
    doc_id = 1  # <-- increments only when a file ends

    # Iterate over files inside the directory
    for archivo in os.listdir(DIRECTORIO):

        if archivo.lower().endswith(".txt"):
            ruta = os.path.join(DIRECTORIO, archivo)
            print(f"Procesando: {archivo}")

            # File name without extension → used as title
            title = os.path.splitext(archivo)[0]

            # Extract paragraphs
            parrafos = procesar_archivo(ruta)

            # Store each paragraph as a separate row
            # but all paragraphs from this file share the same doc_id
            for p in parrafos:
                filas.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": p
                })

            # Increase doc_id after finishing this file
            doc_id += 1

    # Convert list of rows into a DataFrame
    df = pd.DataFrame(filas)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV generado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
