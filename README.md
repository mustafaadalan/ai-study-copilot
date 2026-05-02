# 📚 AI Study Copilot

**AI Study Copilot**, PDF ders notlarını analiz ederek öğrenmeyi hızlandıran yapay zeka destekli bir çalışma asistanıdır. Pasif okuma yerine aktif tekrar sağlar: soru sor, cevap al, quiz çöz, zayıf konularını keşfet.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| 🔍 **Anlamsal Arama** | Bi-Encoder + Cross-Encoder ile iki aşamalı FAISS tabanlı retrieval |
| 🤖 **RAG Cevap Üretimi** | Ollama LLM ile kaynaklara dayalı, halüsinasyon-düşük cevaplar |
| 🧠 **Ollama Quiz** | LLM ile kavrama, analiz ve eleştirel düşünme soruları |
| ⚡ **Hızlı Cloze Quiz** | Keyword extraction ile anında boşluk doldurma soruları |
| 🎯 **Zorluk Seviyesi** | Kolay / Orta / Zor seçeneği ile kişiselleştirilmiş quiz |
| 📊 **Zayıf Konu Analizi** | Hangi bölümlerde zorlandığını takip et, o bölümden tekrar quiz üret |
| 🗂️ **Bölüm Filtresi** | PDF içindeki başlıkları otomatik algıla, sadece istediğin bölümü çalış |
| 💡 **İpucu Sistemi** | Her soru için kaynak cümleye erişim |

---

## Teknik Mimari

```
PDF
 └─► pdf_reader.py       → Sayfa bazlı metin çıkarma (pypdf)
      └─► chunker.py     → Cümle bütünlüklü semantic chunk'lar + bölüm algılama
           └─► retriever.py   → FAISS index + Bi-Encoder + Cross-Encoder re-ranking
                └─► rag.py          → Ollama LLM ile RAG cevap üretimi
                └─► quiz_generator.py → Ollama veya Cloze ile quiz üretimi
                     └─► app.py     → Streamlit arayüzü
```

---

## Kurulum

### Gereksinimler
- Python 3.10+
- [Ollama](https://ollama.com) (RAG ve quiz için)

### 1. Bağımlılıkları yükle

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Ollama model indir (isteğe bağlı)

```bash
ollama pull llama3.1:8b
```

> Ollama olmadan da çalışır; RAG ve Ollama quiz özellikleri devre dışı kalır, hızlı cloze quiz aktif olur.

### 3. Uygulamayı başlat

```bash
streamlit run app.py
```

---

## Kullanım

1. Sol panelden **PDF yükle**
2. **Bölüm filtresi** ile odaklanmak istediğin konuyu seç
3. **Kaynak Arama** sekmesinde soru sor → RAG destekli cevap al
4. **Quiz** sekmesinde soru üret → cevapla → skorunu gör
5. Yanlış yaptığın bölümleri **Zayıf Konu Analizi** ile tespit et

---

## Tech Stack

- **Streamlit** — Web arayüzü
- **sentence-transformers** — `all-MiniLM-L6-v2` Bi-Encoder, `ms-marco-MiniLM-L-6-v2` Cross-Encoder
- **FAISS** — Vektör indeksleme ve arama
- **pypdf** — PDF metin çıkarma
- **Ollama** — Yerel LLM (llama3.1:8b veya başka model)
- **MySQL** — Oturum/kullanıcı verisi için opsiyonel veritabanı desteği

---

## Proje Yapısı

```
ai-study-copilot/
├── app.py                  # Ana Streamlit uygulaması
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Tema ve sunucu ayarları
├── src/
│   ├── pdf_reader.py       # PDF okuma
│   ├── chunker.py          # Metin parçalama
│   ├── retriever.py        # FAISS + embedding arama
│   ├── rag.py              # Ollama RAG entegrasyonu
│   ├── quiz_generator.py   # Quiz üretimi (Ollama + Cloze)
│   └── db.py               # MySQL bağlantı yönetimi
└── data/
    └── sample.pdf          # Örnek PDF
```
