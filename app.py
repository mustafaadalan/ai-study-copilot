import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from difflib import SequenceMatcher

from src.pdf_reader import read_pdf
from src.chunker import chunk_pages
from src.retriever import build_index, search_chunks
from src.quiz_generator import generate_quiz_from_chunks
from src.rag import generate_grounded_answer

st.set_page_config(page_title="AI Study Copilot", layout="wide")

st.title("AI Study Copilot")
st.write("PDF yükle, arama yap veya quiz üret.")


@st.cache_resource
def load_models():
    # Hem Bi-Encoder hem Cross-Encoder modellerini yüklüyoruz
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi_encoder, cross_encoder


def build_answer(results):
    if not results:
        return "Uygun bir cevap bulunamadı.", []

    combined_text = " ".join([item["chunk"] for item in results])
    pages = sorted(list(set(item["page"] for item in results)))
    answer = combined_text[:400]

    return answer, pages


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_query(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def find_heading_matches(query, chunks):
    query_norm = normalize_query(query)
    if not query_norm:
        return []

    matched_chunks = []
    for item in chunks:
        section_norm = normalize_query(item["section"])
        if not section_norm:
            continue

        # Exact or near-exact section title matches get priority.
        if query_norm == section_norm or query_norm in section_norm or section_norm in query_norm:
            matched_chunks.append(item)

    return matched_chunks


def get_relevant_results(query, model, index, active_chunks, cross_encoder):
    heading_matches = find_heading_matches(query, active_chunks)
    if heading_matches:
        local_index = build_index(heading_matches, model)
        local_top_k = min(4, len(heading_matches))
        heading_results = search_chunks(
            query,
            model,
            local_index,
            heading_matches,
            cross_encoder=cross_encoder,
            top_k=local_top_k
        )
        return heading_results, True

    semantic_top_k = min(4, len(active_chunks))
    semantic_results = search_chunks(
        query,
        model,
        index,
        active_chunks,
        cross_encoder=cross_encoder,
        top_k=semantic_top_k
    )
    return semantic_results, False


def is_answer_correct(user_answer, true_answer):
    user_norm = normalize_answer(user_answer)
    true_norm = normalize_answer(true_answer)

    if not user_norm:
        return False

    if user_norm == true_norm:
        return True

    if user_norm in true_norm or true_norm in user_norm:
        return True

    similarity = SequenceMatcher(None, user_norm, true_norm).ratio()
    return similarity >= 0.80


def init_quiz_state():
    if "current_quiz_items" not in st.session_state:
        st.session_state.current_quiz_items = []
    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = {}
    if "graded_questions" not in st.session_state:
        st.session_state.graded_questions = set()
    if "wrong_by_section" not in st.session_state:
        st.session_state.wrong_by_section = {}
    if "wrong_by_page" not in st.session_state:
        st.session_state.wrong_by_page = {}
    if "quiz_round" not in st.session_state:
        st.session_state.quiz_round = 0


def update_weak_stats(item, is_correct):
    question_id = f"{item['section']}|{item['page']}|{item['question']}"
    if question_id in st.session_state.graded_questions:
        return

    st.session_state.graded_questions.add(question_id)

    if is_correct:
        return

    section = item["section"]
    page = item["page"]

    st.session_state.wrong_by_section[section] = (
        st.session_state.wrong_by_section.get(section, 0) + 1
    )
    st.session_state.wrong_by_page[page] = (
        st.session_state.wrong_by_page.get(page, 0) + 1
    )


# Modelleri yüklüyoruz
model, cross_encoder = load_models()
init_quiz_state()

uploaded_file = st.file_uploader("PDF yükle", type=["pdf"])
use_ollama_rag = st.checkbox("Ollama ile RAG cevap üret", value=True)
ollama_model_name = st.text_input("Ollama model adı", value="llama3.1:8b")

if uploaded_file is not None:
    pages = read_pdf(uploaded_file)
    chunks = chunk_pages(pages, max_chunk_words=120, overlap_sentences=1)

    available_sections = sorted(
        list(set(item["section"] for item in chunks if item["section"] != "Genel"))
    )

    selected_section = st.selectbox(
        "Arama / quiz alanı seç:",
        ["Tüm PDF"] + available_sections
    )

    if selected_section == "Tüm PDF":
        active_chunks = chunks
    else:
        active_chunks = [item for item in chunks if item["section"] == selected_section]

    st.success(
        f"PDF işlendi. Toplam sayfa: {len(pages)} | "
        f"Toplam chunk: {len(chunks)} | "
        f"Aktif chunk: {len(active_chunks)}"
    )

    if len(active_chunks) > 0:
        index = build_index(active_chunks, model)
    else:
        index = None

    tab1, tab2 = st.tabs(["Kaynak Arama", "Quiz Üret"])

    with tab1:
        query = st.text_input("Sorunu yaz:")

        if st.button("Ara / Cevapla", key="search_button"):
            if query.strip() == "":
                st.warning("Lütfen bir soru gir.")
            elif index is None:
                st.error("Bu seçimde aranacak chunk bulunamadı.")
            else:
                results, is_heading_match = get_relevant_results(
                    query,
                    model,
                    index,
                    active_chunks,
                    cross_encoder
                )
                source_preview, source_pages = build_answer(results)
                answer = source_preview

                if use_ollama_rag:
                    rag_answer, rag_error = generate_grounded_answer(
                        query=query,
                        results=results,
                        model_name=ollama_model_name.strip() or "llama3.1:8b"
                    )
                    if rag_error:
                        st.warning(f"RAG cevabı üretilemedi, kaynak önizleme gösteriliyor. ({rag_error})")
                    elif rag_answer:
                        answer = rag_answer

                st.subheader("Ön İzleme Cevap")
                st.write(answer)
                if is_heading_match:
                    st.caption("Soru başlık eşleşmesi bulundu; cevap o başlığa ait PDF parçalarından üretildi.")

                st.subheader("Kaynak Sayfalar")
                st.write(", ".join(str(page) for page in source_pages))

                with st.expander("Bulunan kaynak parçaları"):
                    for i, item in enumerate(results, start=1):
                        st.markdown(f"### Sonuç {i}")
                        st.write(f"**Bölüm:** {item['section']}")
                        st.write(f"**Sayfa:** {item['page']}")
                        # Cross encoder skoru varsa onu da göster
                        score_text = f"Bi-Encoder Skor: {item.get('retrieval_score', item.get('score', 0)):.4f}"
                        if 'cross_score' in item:
                            score_text += f" | Cross-Encoder Skor: {item['cross_score']:.4f}"
                        st.write(f"**Skor:** {score_text}")
                        st.write(item["chunk"][:300] + "...")
                        st.divider()

    with tab2:
        quiz_count = st.slider(
            "Kaç quiz üretilsin?",
            min_value=3,
            max_value=10,
            value=5
        )

        if st.button("Quiz Üret", key="quiz_button"):
            if len(active_chunks) == 0:
                st.error("Bu seçimde quiz üretilecek chunk bulunamadı.")
            else:
                st.session_state.current_quiz_items = generate_quiz_from_chunks(
                    active_chunks,
                    limit=quiz_count
                )
                st.session_state.quiz_feedback = {}
                st.session_state.graded_questions = set()
                st.session_state.quiz_round += 1

                if not st.session_state.current_quiz_items:
                    st.warning("Quiz üretilemedi.")
                else:
                    st.rerun()

        if st.session_state.current_quiz_items:
            st.subheader("Quizler")
            st.caption("Önce cevabını yaz, sonra kontrol et.")

            for i, item in enumerate(st.session_state.current_quiz_items, start=1):
                st.markdown(f"### Quiz {i}")
                st.write(item["question"])

                input_key = f"user_answer_{st.session_state.quiz_round}_{i}"
                check_key = f"check_answer_{st.session_state.quiz_round}_{i}"
                feedback_key = f"feedback_{st.session_state.quiz_round}_{i}"

                user_answer = st.text_input("Cevabın:", key=input_key)

                if st.button("Cevabı Kontrol Et", key=check_key):
                    if not user_answer.strip():
                        st.warning("Lütfen önce bir cevap yaz.")
                    else:
                        correct = is_answer_correct(user_answer, item["answer"])
                        st.session_state.quiz_feedback[feedback_key] = {
                            "is_correct": correct,
                            "correct_answer": item["answer"],
                            "user_answer": user_answer
                        }
                        update_weak_stats(item, correct)

                feedback = st.session_state.quiz_feedback.get(feedback_key)
                if feedback:
                    if feedback["is_correct"]:
                        st.success(f"Doğru cevap! Cevabın: {feedback['user_answer']}")
                    else:
                        st.error(f"Yanlış cevap. Senin cevabın: {feedback['user_answer']}")
                        st.info(f"Doğru cevap: {feedback['correct_answer']}")

                st.divider()

            st.subheader("Weak Topic Analizi")
            total_graded = len(st.session_state.quiz_feedback)
            total_correct = sum(
                1 for item in st.session_state.quiz_feedback.values() if item["is_correct"]
            )
            if total_graded > 0:
                success_rate = (total_correct / total_graded) * 100
                st.write(f"Toplam çözülen soru: {total_graded} | Doğruluk oranı: %{success_rate:.1f}")

            if st.session_state.wrong_by_section:
                sorted_sections = sorted(
                    st.session_state.wrong_by_section.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                weak_section, weak_count = sorted_sections[0]
                st.write(f"En çok zorlandığın bölüm: **{weak_section}** ({weak_count} yanlış)")

                if st.button("Zayıf bölümden yeni quiz üret", key="weak_quiz_button"):
                    weak_chunks = [c for c in chunks if c.get("section") == weak_section]
                    if not weak_chunks:
                        st.warning("Bu bölüm için yeterli içerik bulunamadı.")
                    else:
                        st.session_state.current_quiz_items = generate_quiz_from_chunks(
                            weak_chunks,
                            limit=quiz_count
                        )
                        st.session_state.quiz_feedback = {}
                        st.session_state.graded_questions = set()
                        st.session_state.quiz_round += 1
                        st.rerun()
            else:
                st.write("Henüz yanlış cevap verisi oluşmadı.")

            if st.session_state.wrong_by_page:
                sorted_pages = sorted(
                    st.session_state.wrong_by_page.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                page_list = ", ".join(str(page) for page, _ in sorted_pages)
                st.write(f"Tekrar etmen gereken sayfalar: {page_list}")