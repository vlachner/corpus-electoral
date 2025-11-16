import os
import re
import pandas as pd

# ==========================
# CONFIGURACIÓN
# ==========================
DIRECTORIO = "docs-PDFBot/2026"   # <--- CAMBIAR
OUTPUT_CSV = "corpus_parrafos.csv"

# Detecta doble salto de línea con espacios invisibles entre medio
DOUBLE_NL_PATTERN = re.compile(r'\n[ \t]*\n')

def procesar_archivo(ruta):
    """
    Devuelve una lista de párrafos limpios del archivo.
    """
    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    # Normalizar saltos de línea
    contenido = contenido.replace("\r\n", "\n").replace("\r", "\n")

    # Separar por párrafos
    parrafos = DOUBLE_NL_PATTERN.split(contenido)

    parrafos_limpios = []
    for p in parrafos:
        # Unir líneas dentro del mismo párrafo
        p_limpio = re.sub(r'(?<!\n)\n[ \t]*(?!\n)', ' ', p)
        p_limpio = re.sub(r'[ \t]+', ' ', p_limpio).strip()

        if p_limpio:
            parrafos_limpios.append(p_limpio)

    return parrafos_limpios


def main():
    filas = []
    doc_id = 1  # <-- este doc_id solo cambia por archivo

    for archivo in os.listdir(DIRECTORIO):
        if archivo.lower().endswith(".txt"):
            ruta = os.path.join(DIRECTORIO, archivo)
            print(f"Procesando: {archivo}")

            title = os.path.splitext(archivo)[0]
            parrafos = procesar_archivo(ruta)

            # cada párrafo tiene el MISMO doc_id
            for p in parrafos:
                filas.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": p
                })

            doc_id += 1  # <-- aumenta solo cuando termina el archivo

    df = pd.DataFrame(filas)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV generado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
