import os
import fitz  # PyMuPDF
import re
import spacy

def parse_path_for_metadata(fpath):
    """
    Given a path like 'docs/FA/FA-PG-2018.pdf',
    extract (author, doc_type, year).
    """
    filename = os.path.splitext(os.path.basename(fpath))[0]
    parts = filename.split("-")

    author, doc_type, year = None, None, None

    if len(parts) >= 3:
        author = parts[0]
        doc_type = parts[1]
        year = parts[2]

    return author, doc_type, year

def detectPdfType(path_pdf):
    """
    Analiza 3 páginas distribuidas (25%, 50% y 75%) del PDF
    para decidir el mejor método de extracción:
        - "texto_simple"   → usar pdfplumber
        - "texto_complejo" → usar PyMuPDF
        - "imagen"         → usar OCR
    """
    try:
        doc = fitz.open(path_pdf)
        n = len(doc)
        if n == 0:
            return "desconocido"

        # Páginas representativas (25%, 50%, 75%)
        sample_indices = sorted(set([
            max(0, int(n * 0.25) - 1),
            max(0, int(n * 0.5) - 1),
            max(0, int(n * 0.75) - 1)
        ]))

        text_lengths = []
        avg_widths = []
        var_widths = []
        short_line_ratios = []

        for i in sample_indices:
            page = doc.load_page(i)
            text = page.get_text("text").strip()

            # Página casi vacía → probablemente imagen
            if len(text) < 30:
                text_lengths.append(0)
                continue

            blocks = page.get_text("blocks")
            if not blocks:
                continue

            widths = [(b[2] - b[0]) for b in blocks]
            avg_width = sum(widths) / len(widths)
            var_width = sum((w - avg_width) ** 2 for w in widths) / len(widths)
            page_width = page.rect.width

            # Métricas de página
            text_lengths.append(len(text))
            avg_widths.append(avg_width / page_width)
            var_widths.append(var_width)

            # Densidad de líneas cortas
            lines = text.split("\n")
            short_lines = sum(1 for l in lines if len(l.strip()) < 40)
            short_line_ratios.append(short_lines / len(lines) if lines else 0)

        # Si la mayoría de páginas tienen poco texto → OCR
        if sum(t == 0 for t in text_lengths) >= 2:
            return "imagen"

        # Promedios globales
        rel_avg_width = sum(avg_widths) / len(avg_widths)
        rel_var_width = sum(var_widths) / len(var_widths)
        short_line_ratio = sum(short_line_ratios) / len(short_line_ratios)

        # --- Heurísticas ---
        if (rel_avg_width < 0.6 and short_line_ratio > 0.25) or rel_var_width > 1e5:
            return "texto_complejo"
        else:
            return "texto_simple"

    except Exception as e:
        print(f"[Error detectando tipo PDF] {e}")
        return "desconocido"

def is_irrelevant_sentence(sentence):
    sentence = sentence.strip()

    # 1. Oraciones muy cortas
    if len(sentence.split()) < 5:
        return True

    # 2. Notas de página
    if re.match(r'^(P\.|Pág\.|pág\.)', sentence):
        return True

    # 3. Referencias con año entre paréntesis o punto seguido de año
    if re.search(r'\(\d{4}\)|\b\d{4}\b', sentence):
        # Filtramos si parece parte de bibliografía
        # Excepción: si la oración tiene contenido significativo al inicio
        if not re.search(r'\b(el|la|los|las|un|una|es|son)\b', sentence.lower()):
            return True

    # 4. Palabras clave de bibliografía
    if re.search(r'\b(edición|ed\.|cap[s]?\.|pp\.|Editorial|Universidad|Centro|Instituto|Fundación|Buenos Aires|Madrid|San José)\b', sentence, re.IGNORECASE):
        return True

    return False

def filter_paragraphs(paragraphs, pgNumber):
    filtered_paragraphs = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?…])\s+', para)
        kept = [s.strip() for s in sentences if not is_irrelevant_sentence(s)]
        if kept:
            filtered_paragraphs.append((' '.join(kept), pgNumber))
    return filtered_paragraphs

def clean_pdf_text(text):
    """
    Limpieza avanzada del texto extraído de un PDF:
    - Normaliza saltos de línea.
    - Elimina encabezados o bloques iniciales antes del primer doble salto (solo si está muy cerca del inicio).
    - Une palabras partidas y letras separadas.
    - Elimina encabezados y números de página.
    - Mantiene saltos de línea entre párrafos normales.
    """
    if not text:
        return ""

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Normalizar saltos de línea
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 🔹 Quitar espacios entre saltos de línea (" \n \n " → "\n\n")
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)

    # ✅ Eliminar encabezados iniciales solo si el doble salto está al principio (ej. primeras 300 chars)
    match = re.search(r'\n{2,}', text)
    if match and match.start() < 150:
        text = text[match.end():]

    # Unir palabras partidas con guion al final de línea
    text = re.sub(r'-\n', '', text)

    # Unir letras separadas por espacios
    text = re.sub(r'\b(?:[a-zA-Z]\s)+[a-zA-Z]\b',
                  lambda m: m.group(0).replace(' ', ''), text)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        if re.fullmatch(r'[\d\s\-]+', line_strip):
            continue
        if line_strip.upper() in [
            "INTRODUCCIÓN", "CAPÍTULO", "TÍTULO", "SECCIÓN", "ANEXO", "ÍNDICE"
        ]:
            continue
        if line_strip.isupper() and len(line_strip.split()) <= 5:
            continue
        cleaned_lines.append(line_strip)

    return '\n'.join(cleaned_lines)

def is_title_page(text):
    if not text:
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 12:
        return False
    upper_lines = sum(1 for l in lines if len(l) > 2 and l.upper() == l)
    upper_ratio = upper_lines / len(lines)
    if upper_ratio < 0.6:
        return False

    # Solo aplicar revisión de verbos si menos del 80% de líneas son mayúsculas
    if upper_ratio < 0.8:
        doc = nlp(" ".join(lines))
        if any(tok.pos_ == "VERB" for tok in doc):
            return False

    return True

def is_index_page(text, page_blocks=None):
    """
    Heurística para detectar si una página es un índice o tabla de contenido.

    Parámetros:
    - text: texto completo de la página
    - page_blocks: lista opcional de bloques (PyMuPDF) para análisis más preciso

    Retorna:
    - True si se detecta que la página es un índice, False en caso contrario
    """
    if not text:
        return False

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    num_lines = len(lines)
    upper = text.upper()

    # --- 1️ Buscar título típico de índice en las primeras líneas o primer bloque ---
    index_keywords = ["ÍNDICE", "INDEX", "CONTENIDO", "TABLE OF CONTENTS"]
    lines_to_check = lines[:5]  # primeras 5 líneas

    for l in lines_to_check:
        l_clean = l.strip().upper()
        # línea corta (menos de 10-12 caracteres) y coincide con palabra clave → índice
        if len(l_clean) <= 12 and any(word == l_clean for word in index_keywords):
            print("Se detectó título típico de índice en primeras líneas")
            return True

    # --- 2️ Detectar patrón de líneas con puntos y números al final ---
    index_lines = sum(1 for l in lines if re.search(r"\.{3,}\s*\d+$", l))
    if index_lines > num_lines * 0.3:  # más del 30% de líneas
        print("Se detectó patrón de índice por puntos y números al final")
        return True

    # --- 3️ Evitar falsos positivos: ignorar si hay texto relevante suficiente ---
    if num_lines > 20:
        # buscar verbos o sustantivos con spaCy solo si nlp está disponible
        try:
            import spacy
            nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])
            doc = nlp(text)
            if any(tok.pos_ in ["VERB", "NOUN"] for tok in doc):
                return False
        except Exception:
            pass  # si falla spaCy, se ignora esta regla

    # --- 4️ Si ninguna regla se activó, no es índice ---
    return False

def split_paragraphs(text):
    """
    Separa el texto limpio en párrafos:
    - Une líneas partidas dentro de un párrafo.
    - Detecta párrafos por doble salto de línea o punto seguido de mayúscula.
    - Separa viñetas en párrafos individuales.
    """
    # Unir saltos de línea simples (líneas partidas) en un solo espacio
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Dividir párrafos: doble salto de línea o punto seguido de mayúscula
    paragraphs = re.split(r'\n{2,}|(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑ])', text)
    
    final_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Detectar viñetas y separarlas en párrafos individuales
        if '•' in para or '-' in para or '*' in para:
            # Separa por viñeta y limpia espacios
            bullets = [b.strip() for b in re.split(r'[•\-*]', para) if b.strip()]
            final_paragraphs.extend(bullets)
        else:
            final_paragraphs.append(para)
    
    return final_paragraphs
