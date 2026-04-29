import json
import urllib.error
import urllib.request


def build_context(results):
    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[Kaynak {i}] Sayfa: {item['page']} | Bolum: {item['section']}\n{item['chunk']}"
        )
    return "\n\n".join(blocks)


def generate_grounded_answer(query, results, model_name="llama3.1:8b"):
    if not results:
        return None, "Cevap uretmek icin kaynak bulunamadi."

    context = build_context(results)
    system_prompt = (
        "Sen bir PDF calisma asistanisin. Sadece verilen kaynak metinlere dayanarak cevap ver. "
        "Kaynakta gecmeyen bilgi ekleme. Emin degilsen 'Kaynakta acik bilgi yok' de. "
        "Kisa, net ve ogretici cevap ver."
    )
    user_prompt = (
        f"Soru:\n{query}\n\n"
        f"Kaynak Metinler:\n{context}\n\n"
        "Yanit formati:\n"
        "- Cevap: <kisa aciklama>\n"
        "- Dayanak: <kullandigin Kaynak numaralari>"
    )

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.1
        }
    }

    request = urllib.request.Request(
        url="http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            message = body.get("message", {})
            content = message.get("content", "").strip()
            if not content:
                return None, "Ollama bos yanit dondu."
            return content, None
    except urllib.error.URLError as err:
        return None, f"Ollama baglanti hatasi: {err}"
    except Exception as err:
        return None, f"Ollama hatasi: {err}"
