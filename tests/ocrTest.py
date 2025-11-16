import os
import re
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
from textblob import TextBlob
from unidecode import unidecode
import language_tool_python

tool = language_tool_python.LanguageTool('es')

# ---------- CONFIGURACIÓN ----------
PDF_PATH = "docs/2026/PAC.pdf"
OUTPUT_CSV = "PAC_2026.csv"
AUTHOR = "PAC"
DOC_TYPE = "PG"
YEAR = 2026
PDF_PATH_FIELD = "docs/2026/PAC.pdf"

# Define el rango de páginas que quieres procesar (1-indexed)
START_PAGE = 8   # primera página a procesar
END_PAGE = 207    # última página a procesar

# ---------- FUNCIONES AUXILIARES ----------

def clean_text(text):
    """Normaliza texto OCR: elimina errores, espacios, encabezados."""
    text = unidecode(text)  # quitar tildes malformadas
    text = text.replace("-\n", "")  # unir cortes de línea con guiones
    text = re.sub(r"\n+", "\n", text)  # colapsar múltiples saltos
    text = re.sub(r"\s+", " ", text).strip()

    # Eliminar encabezados/títulos comunes
    patterns = [
        r"\bINTRODUCCION\b", r"\bPLAN DE GOBIERNO\b", r"\bPROPUESTAS\b",
        r"\bEJE TEMATICO\b", r"\bOBJETIVO\b", r"\bENFOQUES TRANSVERSALES\b",
        r"\bPARTIDO FRENTE AMPLIO\b", r"^\d+$"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    return text.strip()

def correct_text(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def extract_paragraphs(text):
    """
    Divide el texto OCR en párrafos basándose en doble salto de línea,
    longitud máxima de oraciones y puntuación.
    """
    # Si no hay dobles saltos, intentamos detectar fin de párrafo por punto y seguido.
    if "\n\n" not in text:
        # Dividir por puntos y restaurar coherencia
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', text)
        paragraphs = []
        temp = ""
        for part in parts:
            temp += part.strip() + " "
            # cada ~3 oraciones o más de 400 caracteres, cortamos como párrafo
            if len(temp) > 400 or temp.count('.') >= 3:
                paragraphs.append(temp.strip())
                temp = ""
        if temp.strip():
            paragraphs.append(temp.strip())
    else:
        # Si sí hay dobles saltos, usamos el método original
        paragraphs = [p.strip().replace("\n", " ") for p in text.split("\n\n") if len(p.strip()) > 20]

    return paragraphs


# ---------- PROCESAMIENTO PRINCIPAL ----------

def process_pdf(pdf_path, start_page=None, end_page=None):
    print(f"Procesando {pdf_path} (páginas {start_page}-{end_page}) ...")
    images = convert_from_path(pdf_path, dpi=300, first_page=start_page, last_page=end_page)
    data = []

    for idx, img in enumerate(images, start=start_page):
        print(f"→ Página {idx}/{end_page} ...")
        ocr_text = pytesseract.image_to_string(img, lang='spa')
        ocr_text = clean_text(ocr_text)
        paragraphs = extract_paragraphs(ocr_text)

        for p in paragraphs:
            try:
                corrected = correct_text(p)
            except Exception:
                corrected = p  # fallback si falla el corrector

            data.append({
                "author": AUTHOR,
                "document_type": DOC_TYPE,
                "year": YEAR,
                "sentence": corrected,
                "pdf_path": PDF_PATH_FIELD,
                "page_number": idx
            })

    df = pd.DataFrame(data)
    print(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ OCR completado. Archivo generado: {OUTPUT_CSV}")

# ---------- EJECUCIÓN ----------
if __name__ == "__main__":
    process_pdf(PDF_PATH, start_page=START_PAGE, end_page=END_PAGE)
