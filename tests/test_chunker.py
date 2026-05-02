from src.chunker import chunk_pages, split_into_sentences, detect_major_section_title


SAMPLE_PAGES = [
    {"page": 1, "text": "Pointers store memory addresses. A pointer variable holds the address of another variable. This is useful for dynamic memory allocation."},
    {"page": 2, "text": "POINTER ARITHMETIC\nPointers can be incremented or decremented. Adding 1 to a pointer moves it to the next element. This depends on the data type size."},
    {"page": 3, "text": "Arrays and pointers are closely related in C++. The array name acts as a pointer to the first element. You can use pointer arithmetic to iterate arrays."},
]


def test_chunk_pages_returns_list():
    chunks = chunk_pages(SAMPLE_PAGES)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_keys():
    chunks = chunk_pages(SAMPLE_PAGES)
    for chunk in chunks:
        assert "page" in chunk
        assert "section" in chunk
        assert "chunk" in chunk


def test_chunk_not_empty():
    chunks = chunk_pages(SAMPLE_PAGES)
    for chunk in chunks:
        assert chunk["chunk"].strip() != ""


def test_max_chunk_words():
    chunks = chunk_pages(SAMPLE_PAGES, max_chunk_words=20)
    for chunk in chunks:
        word_count = len(chunk["chunk"].split())
        assert word_count <= 40, f"Chunk çok uzun: {word_count} kelime"


def test_section_detection():
    chunks = chunk_pages(SAMPLE_PAGES)
    sections = {c["section"] for c in chunks}
    assert "POINTER ARITHMETIC" in sections


def test_split_into_sentences_basic():
    text = "Pointers store addresses. They are useful in C++. Memory management depends on them."
    sentences = split_into_sentences(text)
    assert isinstance(sentences, list)


def test_detect_major_section_title_short_text():
    title = detect_major_section_title("POINTER ARITHMETIC")
    assert title == "POINTER ARITHMETIC"


def test_detect_major_section_title_long_text():
    long_text = "This is a very long paragraph that explains pointers in detail. It has multiple sentences and should not be detected as a section title."
    title = detect_major_section_title(long_text)
    assert title is None
