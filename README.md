# AI Study Copilot

AI Study Copilot, PDF ders notlarini analiz ederek ogrenmeyi hizlandirmak icin gelistirilmis mini bir yapay zeka asistanidir.
Proje, dokumani sayfa bazinda okur, metni anlamli parcalara boler, semantik arama ile en ilgili icerikleri bulur ve bu iceriklerden bosluk doldurma (cloze) tipi quiz sorulari uretir.

Bu sayede kullanici sadece metni okumakla kalmaz; ayni zamanda aktif tekrar yaparak konuyu daha kalici sekilde ogrenir.
Kisacasi bu proje, "dokuman okuma" surecini "etkilesimli ogrenme" deneyimine donusturmeyi hedefler.

## Neden Bu Proje?

- Uzun PDF notlarinda dogru bilgiye hizli ulasmak zor.
- Klasik calisma yontemlerinde aktif tekrar eksik kaliyor.
- Ogrenme surecini daha verimli, daha olculebilir ve daha akilli hale getirmek gerekiyor.

## Nasil Calisir?

1. PDF dosyasi okunur (`pdf_reader`).
2. Metin temizlenip chunk'lara ayrilir (`chunker`).
3. Chunk'lar embedding'e cevrilip FAISS index olusturulur (`retriever`).
4. Kullanicinin sorusuna en alakali bolumler bulunur.
5. Icerikten otomatik quiz sorulari uretilir (`quiz_generator`).

## Durum

Proje su anda temel MVP asamasindadir.
Hedef, bu yapiyi adim adim gelistirerek daha guclu bir "study copilot" deneyimine tasimaktir.
