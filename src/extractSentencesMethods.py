import src.utils as utils
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader

def extract_pymupdf_plain_text_page(doc, page_index):
    try:
        page = doc.load_page(page_index)
        return page.get_text("text") or ""
    except Exception:
        return ""
    
def extract_sentences_w_page(pdf_path, num_pages=None, do_ocr_if_needed=True, ocr_order="pymupdf"):
    # inicializar lectores
    reader = PdfReader(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(reader.pages)
    if num_pages is None:
        num_pages = total_pages

    paragraphs = []

    for i in range(min(num_pages, total_pages)):
        # PyMuPDF
        plainPageText = extract_pymupdf_plain_text_page(doc, i)
        isTitlePage = utils.is_title_page(plainPageText)
        isIndexPage = utils.is_index_page(plainPageText)
        if isTitlePage or isIndexPage:
            print(f'Page number {i + 1} is a' + "a Title Page.\n" if isTitlePage else "an Index page.\n")
            continue
        elif plainPageText == '':
            print(f"Page number {i + 1} is empty... it may be an image!\n")
            continue
        noHeadersPgNumbers = utils.clean_pdf_text(plainPageText)
        pageParagraphs = utils.split_paragraphs(noHeadersPgNumbers)
        pageParagraphsCleanedwNumber = utils.filter_paragraphs(pageParagraphs, i + 1)
        paragraphs.extend(pageParagraphsCleanedwNumber)

    doc.close()
    return paragraphs

def extract_text_ocr(path):
    """Extrae texto de PDFs escaneados (por OCR)."""
    doc = fitz.open(path)
    texto = ""
    for page_index in range(len(doc)):
        pix = doc.load_page(page_index).get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        texto += pytesseract.image_to_string(img, lang="spa") + "\n"
    return texto

def extract_sentences_from_pdf(filepath, debug_pages=20):
    """
    Extrae oraciones relevantes desde un PDF.
    - Detecta el tipo de PDF (simple, complejo, imagen)
    - Ignora páginas de título o índice
    - Devuelve lista de tuplas: (oración, número de página)
    - debug_pages: número de páginas para mostrar texto completo y limpio (None para no mostrar)
    """
    sentences_with_page = []

    # Detectar tipo de PDF
    tipo = utils.detectPdfType(filepath)
    print(f"\n→ {os.path.basename(filepath)} → tipo detectado: {tipo}")
    # Para pruebas puedes forzar tipo:
    # tipo = "texto_complejo"

    try:
        # Abrir PDF
        doc = fitz.open(filepath)
        num_pages = len(doc)

        # --- 1️⃣ Extraer todo el texto del PDF según tipo ---
        if tipo == "texto_simple" or tipo == "texto_complejo":
            sentences_with_page = extract_sentences_w_page(filepath)
        elif tipo == "imagen":
            sentences_with_page = extract_text_ocr(filepath)
        else:
            print(f"⚠️ Tipo de PDF desconocido: {filepath}")
            return []

    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")

    return sentences_with_page