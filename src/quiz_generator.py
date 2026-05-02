import json
import re
import urllib.error
import urllib.request
from pathlib import Path


BASE_STOPWORDS = {
    "ve", "veya", "ile", "olarak", "olan", "olup", "gibi", "daha", "çok",
    "bir", "bu", "şu", "da", "de", "için", "en", "göre", "ait",
    "kadar", "sonra", "önce", "olanlar", "üzerinde",
    "altında", "içinde", "nedeniyle", "dolayı", "ancak", "bazen",
    "bölümü", "önemli", "görülen", "alanlarında", "arasında", "üzerine",
    "genelde", "genel", "özellikle", "ayrıca", "sadece", "yaklaşık",
    "olarak", "şekilde", "durumunda", "ancak", "fakat", "çünkü",
    "olarak", "içerir", "edilir", "olduğu", "olduğu", "olmak", "olanın",
    "and", "or", "with", "without", "for", "from", "into", "onto", "over",
    "under", "between", "among", "about", "across", "through", "within",
    "to", "of", "in", "on", "at", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "a", "an", "the", "there", "their", "them", "they", "we", "you", "our",
    "can", "could", "may", "might", "should", "would", "will",
    "also", "often", "generally", "mainly", "mostly", "approximately",
    "important", "section", "chapter", "figure", "table", "example",
    "including", "contains", "consists", "using", "used", "based", "related",
}

NOISY_HINTS = {"Dr.", "Yüksek Mühendisi", "Araştırma Enstitüsü", "Haziran", "Tokat"}

INFORMATIVE_HINTS = (
    "dır", "dir", "dur", "dür",
    "tır", "tir", "tur", "tür",
    "maktadır", "mektedir",
    "olabilir", "olmaktadır", "görülmektedir",
    "sahiptir", "aittir", "cinsindendir", "virüstür",
    "taşınmaktadır", "taşınabilmektedir",
    "kullanılmalıdır", "edilmelidir", "yapılmalıdır",
    "korumaktadır", "oluşturmaktadır",
    "enfekte", "yayılabilmektedir",
)


def load_extra_stopwords(file_path="data/stopwords.txt"):
    path = Path(file_path)
    if not path.exists():
        return set()
    extras = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        word = raw_line.strip().lower()
        if not word or word.startswith("#"):
            continue
        extras.add(word)
    return extras


STOPWORDS = BASE_STOPWORDS | load_extra_stopwords()


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("•", ". ").replace("➢", ". ").replace("►", ". ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def remove_page_markers(text):
    text = re.sub(r"\(?\b\d+\s*/\s*\d+\b\)?", " ", text)
    text = re.sub(r"\b(?:sayfa|page)\s*\d+\b", " ", text, flags=re.IGNORECASE)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped) or re.fullmatch(r"\d+", stripped):
            continue
        lines.append(stripped)
    cleaned = "\n".join(l for l in lines if l)
    return re.sub(r"[ ]{2,}", " ", cleaned).strip()


def is_mostly_uppercase(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if ch.isupper()) / len(letters) > 0.6


def is_informative_sentence(sentence):
    sentence = sentence.strip()
    if len(sentence) < 40 or len(sentence) > 220:
        return False
    words = re.findall(r"\b[\wÇĞİÖŞÜçğıöşü-]+\b", sentence)
    if len(words) < 6:
        return False
    letters = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", sentence)
    digits = re.findall(r"\d", sentence)
    if letters and len(digits) > len(letters) * 0.4:
        return False
    if is_mostly_uppercase(sentence):
        return False
    if any(noisy in sentence for noisy in NOISY_HINTS):
        return False
    return True


def split_into_sentences(text):
    text = normalize_text(text)
    text = remove_page_markers(text)
    raw = re.split(r"(?<=[.!?])\s+", text)
    result = []
    for s in raw:
        s = s.strip(" -•➢.,;:")
        if is_informative_sentence(s):
            result.append(s)
    return result


def find_keyword(sentence):
    words = re.findall(r"\b[\wÇĞİÖŞÜçğıöşü-]+\b", sentence)
    candidates = []
    for word in words:
        if word.lower() in STOPWORDS or len(word) < 4 or word.isdigit():
            continue
        priority = 0
        upper_score = sum(1 for ch in word if ch.isupper())
        if upper_score >= 2:
            priority = 3
        elif word[:1].isupper():
            priority = 2
        elif len(word) >= 8:
            priority = 1
        candidates.append((priority, len(word), word))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def make_cloze(sentence, keyword):
    return re.sub(r"\b" + re.escape(keyword) + r"\b", "_____", sentence, count=1)


# ── Cloze (fallback) ──────────────────────────────────────────────────────────

def generate_quiz_from_chunks(chunks, limit=5):
    """Boşluk doldurma yöntemiyle hızlı quiz üretir. Ollama başarısız olursa fallback."""
    seen = set()
    items = []
    for item in chunks:
        section = item.get("section", "Genel")
        page = item.get("page", "?")
        for sentence in split_into_sentences(item.get("chunk", "")):
            keyword = find_keyword(sentence)
            if not keyword:
                continue
            question = make_cloze(sentence, keyword)
            if question == sentence or question in seen:
                continue
            seen.add(question)
            items.append({
                "section": section,
                "page": page,
                "question": question,
                "answer": keyword,
                "source_sentence": sentence,
                "type": "cloze",
            })
            if len(items) >= limit:
                return items
    return items


# ── Ollama (kaliteli, batch) ──────────────────────────────────────────────────

def generate_quiz_with_ollama(chunks, limit=5, model_name="llama3.2:3b", difficulty="orta"):
    """Tek API çağrısında tüm soruları üretir — çok daha hızlı."""
    import random

    difficulty_desc = {
        "kolay": "temel kavramları test eden, kısa cevaplı",
        "orta": "anlama ve analiz gerektiren",
        "zor": "eleştirel düşünme gerektiren, detaylı cevaplı",
    }.get(difficulty, "anlama gerektiren")

    valid = [c for c in chunks if len(c.get("chunk", "").split()) >= 20]
    random.shuffle(valid)
    selected = valid[:min(limit, len(valid))]

    if not selected:
        return []

    context_parts = []
    for i, item in enumerate(selected, 1):
        context_parts.append(f"[Metin {i}]\n{item['chunk'][:350]}")
    context = "\n\n".join(context_parts)

    raw = _ollama_ask_batch(context, model_name, difficulty_desc, limit)

    items = []
    seen = set()
    for i, q in enumerate(raw[:limit]):
        question = q.get("question", "").strip()
        answer = q.get("answer", "").strip()
        if not question or not answer or question in seen:
            continue
        seen.add(question)
        source = selected[min(i, len(selected) - 1)]
        items.append({
            "section": source.get("section", "Genel"),
            "page": source.get("page", "?"),
            "question": question,
            "answer": answer,
            "source_sentence": source.get("chunk", "")[:200],
            "type": "ollama",
        })

    return items


def _ollama_ask_batch(context, model_name, difficulty_desc, count):
    """Tüm soruları tek çağrıda üretir."""
    prompt = (
        f"Aşağıdaki eğitim metinlerinden {difficulty_desc} tam olarak {count} soru üret.\n"
        "Türkçe yaz. Cevaplar kısa ve net olsun.\n"
        f"SADECE {count} elemanlı JSON dizisi döndür, başka hiçbir şey yazma:\n"
        '[{"question": "...", "answer": "..."}, ...]\n\n'
        f"Metinler:\n{context}\n\nJSON:"
    )
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.35, "top_p": 0.9},
    }
    try:
        req = urllib.request.Request(
            url="http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            content = body.get("message", {}).get("content", "").strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
    except Exception:
        pass
    return []


if __name__ == "__main__":
    from src.pdf_reader import read_pdf
    from src.chunker import chunk_pages

    pages = read_pdf("data/sample.pdf")
    chunks = chunk_pages(pages, max_chunk_words=120, overlap_sentences=1)
    quiz_items = generate_quiz_from_chunks(chunks, limit=5)

    for i, item in enumerate(quiz_items, 1):
        print(f"\nQuiz {i} [{item['type']}]")
        print(item["question"])
        print("Cevap:", item["answer"])
