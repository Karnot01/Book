import spacy
import re

# Cargar modelo en español
nlp = spacy.load("es_core_news_sm")

# Leer archivo
with open("Book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- NORMALIZACIÓN ---
def normalize(text):
    text = text.lower()  # minúsculas
    text = re.sub(r"\d+", "", text)  # eliminar números
    text = re.sub(r"[^\w\s]", "", text)  # eliminar puntuación
    text = re.sub(r"\s+", " ", text).strip()  # espacios extra
    return text

clean_text = normalize(text)

# --- LEMATIZACIÓN ---
doc = nlp(clean_text)

lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop])

# Guardar resultado
with open("clean_book.txt", "w", encoding="utf-8") as f:
    f.write(lemmatized)

print("Limpieza completada. Archivo generado: clean_book.txt")