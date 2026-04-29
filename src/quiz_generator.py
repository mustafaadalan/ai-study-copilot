import re
from pathlib import Path


BASE_STOPWORDS = {
    # Turkish
    "ve", "veya", "ile", "olarak", "olan", "olup", "gibi", "daha", "çok",
    "bir", "bu", "şu", "da", "de", "için", "en", "göre", "ait",
    "kadar", "sonra", "önce", "olanlar", "üzerinde",
    "altında", "içinde", "nedeniyle", "dolayı", "ancak", "bazen",
    "bölümü", "önemli", "görülen", "alanlarında", "arasında", "üzerine",
    "genelde", "genel", "özellikle", "ayrıca", "sadece", "yaklaşık",
    "olarak", "şekilde", "durumunda", "ancak", "fakat", "çünkü",
    "olarak", "içerir", "edilir", "olduğu", "olduğu", "olmak", "olanın",
    # English
    "and", "or", "with", "without", "for", "from", "into", "onto", "over",
    "under", "between", "among", "about", "across", "through", "within",
    "to", "of", "in", "on", "at", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "a", "an", "the", "there", "their", "them", "they", "we", "you", "our",
    "can", "could", "may", "might", "should", "would", "will",
    "also", "often", "generally", "mainly", "mostly", "approximately",
    "important", "section", "chapter", "figure", "table", "example",
    "including", "contains", "consists", "using", "used", "based", "related"
}


NOISY_HINTS = {
    "Dr.", "Yüksek Mühendisi", "Araştırma Enstitüsü", "Haziran", "Tokat"
}


INFORMATIVE_HINTS = (
    "dır", "dir", "dur", "dür",
    "tır", "tir", "tur", "tür",
    "maktadır", "mektedir",
    "olabilir", "olmaktadır", "görülmektedir",
    "sahiptir", "aittir", "cinsindendir", "virüstür",
    "taşınmaktadır", "taşınabilmektedir",
    "kullanılmalıdır", "edilmelidir", "yapılmalıdır",
    "korumaktadır", "oluşturmaktadır",
    "enfekte", "yayılabilmektedir"
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
    text = text.replace("•", ". ")
    text = text.replace("➢", ". ")
    text = text.replace("►", ". ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def remove_page_markers(text):
    text = re.sub(r"\(?\b\d+\s*/\s*\d+\b\)?", " ", text)
    text = re.sub(r"\b(?:sayfa|page)\s*\d+\b", " ", text, flags=re.IGNORECASE)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            continue
        if re.fullmatch(r"\d+", stripped):
            continue

        lines.append(stripped)

    cleaned = "\n".join(line for line in lines if line)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()


def is_mostly_uppercase(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    upper_count = sum(1 for ch in letters if ch.isupper())
    ratio = upper_count / len(letters)
    return ratio > 0.6


def is_informative_sentence(sentence):
    sentence = sentence.strip()
    lower_sentence = sentence.lower()

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

    # Keep this as a soft preference instead of a hard rule.
    # Some valid educational sentences may not include these suffixes/phrases.

    return True


def split_into_sentences(text):
    text = normalize_text(text)
    text = remove_page_markers(text)
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip(" -•➢.,;:")
        if is_informative_sentence(sentence):
            clean_sentences.append(sentence)

    return clean_sentences


def find_keyword(sentence):
    words = re.findall(r"\b[\wÇĞİÖŞÜçğıöşü-]+\b", sentence)

    candidates = []
    for word in words:
        lower_word = word.lower()

        if lower_word in STOPWORDS:
            continue

        if len(word) < 4:
            continue

        if word.isdigit():
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
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.sub(pattern, "_____", sentence, count=1)


def generate_quiz_from_chunks(chunks, limit=5):
    seen_questions = set()
    quiz_items = []

    for item in chunks:
        section = item.get("section", "Genel")
        page = item.get("page", "?")
        text = item.get("chunk", "")

        sentences = split_into_sentences(text)

        for sentence in sentences:
            keyword = find_keyword(sentence)
            if not keyword:
                continue

            question = make_cloze(sentence, keyword)

            if question == sentence:
                continue

            if question in seen_questions:
                continue

            quiz_items.append({
                "section": section,
                "page": page,
                "question": question,
                "answer": keyword,
                "source_sentence": sentence
            })

            seen_questions.add(question)

            if len(quiz_items) >= limit:
                return quiz_items

    return quiz_items


if __name__ == "__main__":
    from src.pdf_reader import read_pdf
    from src.chunker import chunk_pages

    pdf_path = "data/sample.pdf"

    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=120, overlap=30)

    quiz_items = generate_quiz_from_chunks(chunks, limit=5)

    for i, item in enumerate(quiz_items, start=1):
        print(f"\nQuiz {i}")
        print(item["question"])
        print("Cevap:", item["answer"])