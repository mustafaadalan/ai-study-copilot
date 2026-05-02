from src.quiz_generator import generate_quiz_from_chunks, find_keyword, make_cloze


SAMPLE_CHUNKS = [
    {
        "page": 1,
        "section": "Pointers",
        "chunk": "A pointer variable stores the memory address of another variable. Pointers are fundamental to dynamic memory allocation in C++.",
    },
    {
        "page": 2,
        "section": "Arrays",
        "chunk": "An array is a collection of elements stored at contiguous memory locations. The array name points to the first element.",
    },
    {
        "page": 3,
        "section": "Functions",
        "chunk": "Functions allow code reuse by encapsulating logic into named blocks. Parameters pass data into the function body.",
    },
    {
        "page": 4,
        "section": "Classes",
        "chunk": "A class defines a blueprint for creating objects. Objects are instances of a class and contain attributes and methods.",
    },
    {
        "page": 5,
        "section": "Inheritance",
        "chunk": "Inheritance allows a derived class to reuse the properties of a base class. This promotes code reusability and polymorphism.",
    },
]


def test_generate_returns_list():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=3)
    assert isinstance(items, list)


def test_generate_respects_limit():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=3)
    assert len(items) <= 3


def test_quiz_item_keys():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=5)
    for item in items:
        assert "question" in item
        assert "answer" in item
        assert "section" in item
        assert "page" in item
        assert "type" in item


def test_cloze_has_blank():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=5)
    cloze_items = [i for i in items if i["type"] == "cloze"]
    for item in cloze_items:
        assert "_____" in item["question"]


def test_cloze_question_differs_from_source():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=5)
    for item in items:
        if item["type"] == "cloze":
            assert item["question"] != item.get("source_sentence", "")


def test_no_duplicate_questions():
    items = generate_quiz_from_chunks(SAMPLE_CHUNKS, limit=10)
    questions = [i["question"] for i in items]
    assert len(questions) == len(set(questions))


def test_find_keyword_returns_string_or_none():
    result = find_keyword("Pointers store memory addresses in C++.")
    assert result is None or isinstance(result, str)


def test_make_cloze_replaces_keyword():
    sentence = "Pointers store memory addresses."
    result = make_cloze(sentence, "Pointers")
    assert "_____" in result
    assert "Pointers" not in result
