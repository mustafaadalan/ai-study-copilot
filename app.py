import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from difflib import SequenceMatcher

from src.pdf_reader import read_pdf
from src.chunker import chunk_pages
from src.retriever import build_index, search_chunks
from src.quiz_generator import generate_quiz_from_chunks, generate_quiz_with_ollama
from src.rag import generate_grounded_answer

st.set_page_config(
    page_title="AI Study Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.71rem;
    font-weight: 700;
    margin-right: 5px;
    margin-bottom: 4px;
    letter-spacing: 0.03em;
}
.badge-purple {
    background: rgba(124,58,237,0.15);
    color: #7c3aed;
    border: 1px solid rgba(124,58,237,0.35);
}
.badge-green {
    background: rgba(16,185,129,0.15);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.35);
}
.badge-blue {
    background: rgba(59,130,246,0.15);
    color: #2563eb;
    border: 1px solid rgba(59,130,246,0.35);
}
.badge-orange {
    background: rgba(249,115,22,0.15);
    color: #ea580c;
    border: 1px solid rgba(249,115,22,0.35);
}
.quiz-card {
    border-left: 4px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 1.3rem 1.5rem 0.8rem;
    margin: 1.2rem 0 0.4rem;
    background: rgba(124,58,237,0.04);
    border-top: 1px solid rgba(124,58,237,0.12);
    border-right: 1px solid rgba(124,58,237,0.12);
    border-bottom: 1px solid rgba(124,58,237,0.12);
}
.quiz-num {
    font-size: 0.72rem;
    font-weight: 800;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}
.quiz-q {
    font-size: 1.05rem;
    font-weight: 500;
    line-height: 1.75;
    margin: 0.4rem 0 0;
}
.result-card {
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.55rem 0;
    background: rgba(59,130,246,0.04);
    border-top: 1px solid rgba(59,130,246,0.12);
    border-right: 1px solid rgba(59,130,246,0.12);
    border-bottom: 1px solid rgba(59,130,246,0.12);
}
.hero {
    text-align: center;
    padding: 5rem 1rem 2.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}
.hero p {
    font-size: 1.1rem;
    opacity: 0.6;
    margin-bottom: 2.8rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.1rem;
    max-width: 660px;
    margin: 0 auto;
}
.feature-card {
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    background: rgba(124,58,237,0.04);
}
.feature-icon { font-size: 1.7rem; margin-bottom: 0.45rem; }
.feature-title { font-size: 0.88rem; font-weight: 700; margin-bottom: 0.25rem; }
.feature-desc { font-size: 0.78rem; opacity: 0.58; }
.score-bar-bg {
    background: rgba(124,58,237,0.1);
    border-radius: 20px;
    height: 10px;
    width: 100%;
    margin: 8px 0 4px;
}
.score-bar-fg {
    background: linear-gradient(90deg, #7c3aed, #3b82f6);
    border-radius: 20px;
    height: 10px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi_encoder, cross_encoder


@st.cache_data(show_spinner="PDF işleniyor...")
def process_pdf(file_bytes):
    import io
    pages = read_pdf(io.BytesIO(file_bytes))
    chunks = chunk_pages(pages, max_chunk_words=120, overlap_sentences=1)
    return pages, chunks


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text)


def find_heading_matches(query, chunks):
    q = normalize(query)
    if not q:
        return []
    return [
        item for item in chunks
        if (sec := normalize(item["section"])) and (q == sec or q in sec or sec in q)
    ]


def get_relevant_results(query, model, index, active_chunks, cross_encoder):
    heading = find_heading_matches(query, active_chunks)
    if heading:
        local_index = build_index(heading, model)
        results = search_chunks(
            query, model, local_index, heading,
            cross_encoder=cross_encoder, top_k=min(4, len(heading))
        )
        return results, True
    results = search_chunks(
        query, model, index, active_chunks,
        cross_encoder=cross_encoder, top_k=min(4, len(active_chunks))
    )
    return results, False


def is_answer_correct(user_answer, true_answer):
    u, t = normalize(user_answer), normalize(true_answer)
    if not u:
        return False
    if u == t:
        return True
    # Doğru cevap kullanıcının uzun cevabı içindeyse kabul et (ör. tam cümle yazdı)
    if t in u and len(u) <= len(t) * 2.5:
        return True
    return SequenceMatcher(None, u, t).ratio() >= 0.85


def init_quiz_state():
    defaults = {
        "current_quiz_items": [],
        "quiz_feedback": {},
        "graded_questions": set(),
        "wrong_by_section": {},
        "wrong_by_page": {},
        "quiz_round": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def update_weak_stats(item, is_correct):
    qid = f"{item['section']}|{item['page']}|{item['question']}"
    if qid in st.session_state.graded_questions:
        return
    st.session_state.graded_questions.add(qid)
    if is_correct:
        return
    st.session_state.wrong_by_section[item["section"]] = (
        st.session_state.wrong_by_section.get(item["section"], 0) + 1
    )
    st.session_state.wrong_by_page[item["page"]] = (
        st.session_state.wrong_by_page.get(item["page"], 0) + 1
    )


model, cross_encoder = load_models()
init_quiz_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 AI Study Copilot")
    st.caption("PDF tabanlı akıllı çalışma asistanı")
    st.divider()

    uploaded_file = st.file_uploader("PDF Yükle", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        st.success(f"📄 {uploaded_file.name}")

    st.divider()
    st.markdown("**⚙️ Ayarlar**")

    use_ollama_rag = st.toggle("RAG ile cevap üret", value=True)
    use_ollama_quiz = st.toggle("Ollama ile quiz üret", value=True)
    ollama_model_name = st.text_input(
        "Ollama model", value="llama3.2:3b",
        label_visibility="collapsed", placeholder="llama3.2:3b"
    )
    difficulty = st.select_slider(
        "Quiz zorluğu",
        options=["kolay", "orta", "zor"],
        value="orta",
    )

# ── Landing page ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="hero">
        <h1>📚 AI Study Copilot</h1>
        <p>PDF yükle, anında sor, quiz üret — yapay zeka destekli çalışma asistanın.</p>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Kaynak Arama</div>
                <div class="feature-desc">Anlamsal arama ve Ollama RAG destekli cevap</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Akıllı Quiz</div>
                <div class="feature-desc">Ollama ile kavrama soruları veya hızlı boşluk doldurma</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Zayıf Konu Analizi</div>
                <div class="feature-desc">Hangi bölümlerde zorlandığını izle, tekrar et</div>
            </div>
        </div>
    </div>
    <br><p style="text-align:center; opacity:0.45;">← Sol panelden PDF yükle</p>
    """, unsafe_allow_html=True)
    st.stop()

# ── PDF işleme (cache'li — her butona basışta yeniden çalışmaz) ───────────────
pages, chunks = process_pdf(uploaded_file.getvalue())
available_sections = sorted(set(c["section"] for c in chunks if c["section"] != "Genel"))

with st.sidebar:
    st.divider()
    st.markdown("**📑 Bölüm Filtresi**")
    selected_section = st.selectbox(
        "Bölüm", ["Tüm PDF"] + available_sections, label_visibility="collapsed"
    )
    st.caption(f"{len(pages)} sayfa · {len(chunks)} chunk")

active_chunks = (
    chunks if selected_section == "Tüm PDF"
    else [c for c in chunks if c["section"] == selected_section]
)

# Index'i session_state'te cache'le — section değişmediği sürece yeniden hesaplanmaz
if "index_cache" not in st.session_state:
    st.session_state.index_cache = {}

_index_key = f"{uploaded_file.name}_{selected_section}"
if _index_key not in st.session_state.index_cache:
    st.session_state.index_cache[_index_key] = (
        build_index(active_chunks, model) if active_chunks else None
    )
index = st.session_state.index_cache[_index_key]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Kaynak Arama", "🧠 Quiz"])

# ── Tab 1: Arama ──────────────────────────────────────────────────────────────
with tab1:
    query = st.text_input("Sorunuzu yazın", placeholder="Örn: Virüs nasıl yayılır?")

    if st.button("Ara / Cevapla", type="primary"):
        if not query.strip():
            st.warning("Lütfen bir soru girin.")
        elif index is None:
            st.error("Seçili bölümde aranacak içerik yok.")
        else:
            with st.spinner("Aranıyor..."):
                results, is_heading = get_relevant_results(
                    query, model, index, active_chunks, cross_encoder
                )

            answer = " ".join(r["chunk"] for r in results)[:400]

            if use_ollama_rag and results:
                with st.spinner("RAG cevabı üretiliyor..."):
                    rag_answer, rag_err = generate_grounded_answer(
                        query=query,
                        results=results,
                        model_name=ollama_model_name.strip() or "llama3.1:8b",
                    )
                if rag_err:
                    st.warning(f"RAG kullanılamadı: {rag_err}")
                elif rag_answer:
                    answer = rag_answer

            st.markdown("#### Cevap")
            st.info(answer)
            if is_heading:
                st.caption("💡 Başlık eşleşmesi bulundu — cevap o bölümden üretildi.")

            pages_found = sorted(set(r["page"] for r in results))
            st.markdown(f"**Kaynak sayfalar:** {', '.join(str(p) for p in pages_found)}")

            with st.expander(f"Kaynak parçaları ({len(results)} sonuç)"):
                for item in results:
                    score = item.get("cross_score", item.get("retrieval_score", 0))
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="badge badge-purple">{item['section']}</span>
                        <span class="badge badge-green">Sayfa {item['page']}</span>
                        <span class="badge badge-blue">Skor {score:.3f}</span>
                        <p style="margin:0.6rem 0 0; line-height:1.6;">{item['chunk'][:300]}…</p>
                    </div>
                    """, unsafe_allow_html=True)

# ── Tab 2: Quiz ───────────────────────────────────────────────────────────────
with tab2:
    col_left, col_right = st.columns([3, 1])
    with col_left:
        quiz_count = st.slider("Kaç soru?", min_value=3, max_value=10, value=5)
    with col_right:
        mode_label = "Ollama" if use_ollama_quiz else "Cloze"
        st.markdown(
            f"<br><span class='badge badge-purple'>Mod: {mode_label} · {difficulty}</span>",
            unsafe_allow_html=True,
        )

    if st.button("Quiz Üret", type="primary"):
        if not active_chunks:
            st.error("Bu bölümde quiz üretilecek içerik yok.")
        else:
            if use_ollama_quiz:
                with st.spinner(f"Ollama ile {quiz_count} soru üretiliyor… (biraz sürebilir)"):
                    items = generate_quiz_with_ollama(
                        active_chunks,
                        limit=quiz_count,
                        model_name=ollama_model_name.strip() or "llama3.1:8b",
                        difficulty=difficulty,
                    )
                if not items:
                    st.warning("Ollama ile soru üretilemedi, boşluk doldurma yöntemine geçiliyor…")
                    items = generate_quiz_from_chunks(active_chunks, limit=quiz_count)
            else:
                items = generate_quiz_from_chunks(active_chunks, limit=quiz_count)

            st.session_state.current_quiz_items = items
            st.session_state.quiz_feedback = {}
            st.session_state.graded_questions = set()
            st.session_state.quiz_round += 1

            if not items:
                st.warning("Quiz üretilemedi.")
            else:
                st.rerun()

    if st.session_state.current_quiz_items:
        st.markdown("---")

        for i, item in enumerate(st.session_state.current_quiz_items, 1):
            input_key = f"ans_{st.session_state.quiz_round}_{i}"
            check_key = f"chk_{st.session_state.quiz_round}_{i}"
            fb_key = f"fb_{st.session_state.quiz_round}_{i}"

            q_type = item.get("type", "cloze")
            type_label = "ollama" if q_type == "ollama" else "boşluk doldurma"
            type_color = "badge-purple" if q_type == "ollama" else "badge-orange"

            st.markdown(f"""
            <div class="quiz-card">
                <div class="quiz-num">Soru {i}</div>
                <span class="badge badge-blue">{item['section']}</span>
                <span class="badge badge-green">Sayfa {item['page']}</span>
                <span class="badge {type_color}">{type_label}</span>
                <p class="quiz-q">{item['question']}</p>
            </div>
            """, unsafe_allow_html=True)

            user_answer = st.text_input(
                "Cevabın:", key=input_key, placeholder="Cevabını buraya yaz…",
                label_visibility="collapsed"
            )

            c1, c2 = st.columns([1, 5])
            with c1:
                check_clicked = st.button("Kontrol Et", key=check_key, type="primary")
            with c2:
                with st.expander("💡 İpucu"):
                    st.caption(item.get("source_sentence", "")[:200])

            if check_clicked:
                if not user_answer.strip():
                    st.warning("Önce bir cevap yaz.")
                else:
                    correct = is_answer_correct(user_answer, item["answer"])
                    st.session_state.quiz_feedback[fb_key] = {
                        "is_correct": correct,
                        "correct_answer": item["answer"],
                        "user_answer": user_answer,
                    }
                    update_weak_stats(item, correct)

            fb = st.session_state.quiz_feedback.get(fb_key)
            if fb:
                if fb["is_correct"]:
                    st.success(f"✅ Doğru! — {fb['user_answer']}")
                else:
                    st.error(f"❌ Yanlış — Senin cevabın: *{fb['user_answer']}*")
                    st.info(f"💡 Doğru cevap: **{fb['correct_answer']}**")

            st.markdown("<br>", unsafe_allow_html=True)

        # ── Skor özeti ────────────────────────────────────────────────────────
        total = len(st.session_state.quiz_feedback)
        correct_count = sum(1 for v in st.session_state.quiz_feedback.values() if v["is_correct"])

        if total > 0:
            st.markdown("---")
            st.markdown("#### Skor Özeti")
            c1, c2, c3 = st.columns(3)
            c1.metric("Çözülen", total)
            c2.metric("Doğru", correct_count)
            c3.metric("Başarı", f"%{correct_count / total * 100:.0f}")
            pct = int(correct_count / total * 100)
            st.markdown(
                f'<div class="score-bar-bg"><div class="score-bar-fg" style="width:{pct}%;"></div></div>',
                unsafe_allow_html=True,
            )

        # ── Zayıf konu ────────────────────────────────────────────────────────
        if st.session_state.wrong_by_section:
            st.markdown("#### Zayıf Konu Analizi")
            top_section, top_count = max(
                st.session_state.wrong_by_section.items(), key=lambda x: x[1]
            )
            st.warning(f"En çok zorlandığın bölüm: **{top_section}** ({top_count} yanlış)")

            if st.button("Bu bölümden yeni quiz üret", key="weak_quiz_btn"):
                weak_chunks = [c for c in chunks if c.get("section") == top_section]
                if not weak_chunks:
                    st.warning("Bu bölüm için yeterli içerik yok.")
                else:
                    if use_ollama_quiz:
                        with st.spinner("Zayıf bölüm için sorular üretiliyor…"):
                            items = generate_quiz_with_ollama(
                                weak_chunks,
                                limit=quiz_count,
                                model_name=ollama_model_name.strip() or "llama3.1:8b",
                                difficulty=difficulty,
                            )
                        if not items:
                            items = generate_quiz_from_chunks(weak_chunks, limit=quiz_count)
                    else:
                        items = generate_quiz_from_chunks(weak_chunks, limit=quiz_count)

                    st.session_state.current_quiz_items = items
                    st.session_state.quiz_feedback = {}
                    st.session_state.graded_questions = set()
                    st.session_state.quiz_round += 1
                    st.rerun()

        if st.session_state.wrong_by_page:
            top_pages = sorted(
                st.session_state.wrong_by_page.items(), key=lambda x: x[1], reverse=True
            )[:5]
            page_list = ", ".join(str(p) for p, _ in top_pages)
            st.info(f"📖 Tekrar etmen gereken sayfalar: **{page_list}**")
