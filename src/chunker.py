import re
from src.pdf_reader import read_pdf

def clean_lines(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines

def is_page_marker(line):
    line = line.strip()
    return bool(
        re.fullmatch(r"\d+\s*/\s*\d+", line)
        or re.fullmatch(r"\d+", line)
    )

def detect_major_section_title(text):
    lines = [line for line in clean_lines(text) if not is_page_marker(line)]

    if not lines:
        return None

    if len(lines) > 3:
        return None

    first = lines[0]

    if first.startswith(("•", "➢", "-", "*")):
        return None

    if len(first) < 4 or len(first) > 90:
        return None

    return first

def split_into_sentences(text):
    """Metni noktalama işaretlerine göre akıllıca cümlelere böler."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # Nokta, soru işareti veya ünlemden sonra boşluk geliyorsa böl
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_pages(pages, max_chunk_words=120, overlap_sentences=1):
    """
    Kelime sayısını körü körüne kesmek yerine, cümle bütünlüğünü koruyarak
    anlamsal (semantic) chunk'lar oluşturur.
    """
    chunks = []
    current_section = "Genel"

    for item in pages:
        page_number = item["page"]
        text = item["text"]

        detected_section = detect_major_section_title(text)
        if detected_section:
            current_section = detected_section

        sentences = split_into_sentences(text)
        
        current_chunk_sentences = []
        current_word_count = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_word_count = len(sentence.split())

            # Eğer tek bir cümle belirlenen limitten büyükse, bütünlüğü bozmamak için tek parça ekle
            if not current_chunk_sentences and sentence_word_count > max_chunk_words:
                chunks.append({
                    "page": page_number,
                    "section": current_section,
                    "chunk": sentence
                })
                i += 1
                continue

            if current_word_count + sentence_word_count <= max_chunk_words:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
                i += 1
            else:
                # Sınır aşıldı, mevcut chunk'ı kaydet
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "page": page_number,
                    "section": current_section,
                    "chunk": chunk_text
                })
                
                # Overlap mekanizması (Bağlam kopmaması için son cümleyi bir sonraki chunk'a devret)
                if overlap_sentences > 0 and len(current_chunk_sentences) > overlap_sentences:
                    i -= overlap_sentences
                    current_chunk_sentences = []
                    current_word_count = 0
                else:
                    current_chunk_sentences = []
                    current_word_count = 0

        # Kalan son parçayı ekle
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "page": page_number,
                "section": current_section,
                "chunk": chunk_text
            })

    return chunks

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Toplam chunk sayısı: {len(chunks)}\n")

    for i, item in enumerate(chunks[:5], start=1):
        print(f"--- Chunk {i} | Sayfa {item['page']} | Bölüm {item['section']} ---")
        print(item["chunk"])
        print()