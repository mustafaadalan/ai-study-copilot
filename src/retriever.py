import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.pdf_reader import read_pdf
from src.chunker import chunk_pages

def build_index(chunks, model):
    chunk_texts = [item["chunk"] for item in chunks]

    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def search_chunks(query, model, index, chunks, cross_encoder=None, top_k=3):
    # 1. Aşama: Dense Retrieval (Bi-Encoder + FAISS) - Geniş bir havuz getir (top 15)
    initial_k = min(15, len(chunks))
    
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(query_embedding, initial_k)

    initial_results = []
    for score, idx in zip(scores[0], indices[0]):
        initial_results.append({
            "page": chunks[idx]["page"],
            "section": chunks[idx]["section"],
            "chunk": chunks[idx]["chunk"],
            "retrieval_score": float(score)
        })

    # Eğer Cross-Encoder tanımlanmamışsa veya havuz zaten çok küçükse doğrudan dön
    if not cross_encoder or len(initial_results) <= top_k:
        return initial_results[:top_k]

    # 2. Aşama: Cross-Encoder ile Re-ranking (Yeniden Sıralama)
    # Soru ve aday metni ikili (pair) olarak modele veriyoruz
    cross_inp = [[query, item["chunk"]] for item in initial_results]
    cross_scores = cross_encoder.predict(cross_inp)

    # Skorları aday listesine ekle
    for i in range(len(initial_results)):
        initial_results[i]["cross_score"] = float(cross_scores[i])

    # Yeni Cross-Encoder skoruna göre listeyi büyükten küçüğe sırala
    ranked_results = sorted(initial_results, key=lambda x: x["cross_score"], reverse=True)

    # Gerçekten en alakalı olanları döndür
    return ranked_results[:top_k]

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print("Embedding modeli (Bi-Encoder) yükleniyor...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Re-ranking modeli (Cross-Encoder) yükleniyor...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("FAISS index oluşturuluyor...")
    index = build_index(chunks, model)

    query = input("Sorunu yaz: ")

    results = search_chunks(query, model, index, chunks, cross_encoder=cross_encoder, top_k=3)

    print("\nEn alakalı chunk'lar:\n")

    for i, item in enumerate(results, start=1):
        score_text = f"Bi-Encoder Skor: {item.get('retrieval_score', 0):.4f}"
        if "cross_score" in item:
            score_text += f" | Cross-Encoder Skor: {item['cross_score']:.4f}"
        print(f"--- Sonuç {i} | Sayfa {item['page']} | {score_text} ---")
        print(item["chunk"])
        print()