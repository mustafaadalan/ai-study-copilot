from pypdf import PdfReader


def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    pages = read_pdf(pdf_path)

    for item in pages:
        print(f"--- Sayfa {item['page']} ---")
        print(item["text"][:500])
        print()