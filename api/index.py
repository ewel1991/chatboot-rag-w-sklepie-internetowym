import os
import sys
import math
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicjalizacja klienta OpenAI (tylko jeśli klucz jest dostępny)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="NeoAsystent RAG API - Lekki Serverless")

# Konfiguracja CORS (umożliwia komunikację z frontendu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalna pamięć podręczna na wektory bazy wiedzy (In-Memory Cache)
# Pozwala uniknąć ponownego generowania wektorów przy ciepłych startach lambdy
cached_chunks_with_embeddings = None


def dot_product(v1, v2):
    """Oblicza iloczyn skalarny dwóch wektorów."""
    return sum(x * y for x, y in zip(v1, v2))


def cosine_similarity(v1, v2):
    """Oblicza podobieństwo cosinusowe dwóch wektorów."""
    # Ponieważ wektory z modelu text-embedding-3-small są już znormalizowane,
    # ich podobieństwo cosinusowe to po prostu iloczyn skalarny.
    return dot_product(v1, v2)


def load_and_embed_knowledge_base():
    """Wczytuje bazę wiedzy, dzieli na bloki i generuje dla nich wektory (Embeddings)."""
    global cached_chunks_with_embeddings

    if cached_chunks_with_embeddings is not None:
        return cached_chunks_with_embeddings

    if not client:
        raise ValueError(
            "Klient OpenAI nie jest zainicjalizowany. Brak klucza API.")

    # Określanie ścieżki do pliku bazy wiedzy
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "knowledge_base_for_RAG.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku bazy wiedzy w ścieżce: {file_path}")

    # Odczyt pliku
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Podział tekstu według separatora
    separator = "===================================================================="
    raw_chunks = content.split(separator)

    chunks = []
    for rc in raw_chunks:
        cleaned = rc.strip()
        if cleaned:
            chunks.append(cleaned)

    if not chunks:
        raise ValueError("Baza wiedzy jest pusta lub źle sformatowana.")

    # Generowanie embeddingów dla wszystkich bloków tekstu za jednym żądaniem API (Batching)
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )

    # Łączenie tekstu z wektorem
    cached_chunks_with_embeddings = []
    for i, data in enumerate(response.data):
        cached_chunks_with_embeddings.append({
            "text": chunks[i],
            "embedding": data.embedding
        })

    print(
        f"Pomyślnie zainicjalizowano bazę RAG: {len(cached_chunks_with_embeddings)} bloków.")
    return cached_chunks_with_embeddings


class ChatRequest(BaseModel):
    message: str


async def execute_chat_logic(message: str):
    """Wspólna logika obsługi zapytania czatu RAG."""
    if not OPENAI_API_KEY or not client:
        return {"response": "Konfiguracja serwera niekompletna. Upewnij się, że dodałeś zmienną OPENAI_API_KEY w panelu Vercel."}

    try:
        # 1. Pobierz lub zainicjalizuj bazę wiedzy z wektorami
        kb_data = load_and_embed_knowledge_base()
    except Exception as e:
        print(f"Błąd ładowania bazy wiedzy: {e}")
        raise HTTPException(
            status_code=500, detail="Nie udało się załadować bazy wiedzy RAG.")

    try:
        # 2. Wygeneruj wektor dla pytania użytkownika
        query_response = client.embeddings.create(
            input=message,
            model="text-embedding-3-small"
        )
        query_vector = query_response.data[0].embedding

        # 3. Przeszukaj bazę wiedzy (Wyszukiwanie Semantyczne)
        scored_chunks = []
        for item in kb_data:
            similarity = cosine_similarity(query_vector, item["embedding"])
            scored_chunks.append((similarity, item["text"]))

        # Sortowanie po najwyższym podobieństwie
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Wybierz 3 najbardziej dopasowane fragmenty
        top_k = scored_chunks[:3]
        context = "\n\n---\n\n".join([text for _, text in top_k])

        # 4. Generowanie odpowiedzi przy użyciu modelu LLM
        system_prompt = (
            "Jesteś NeoAsystentem, profesjonalnym doradcą klienta w sklepie z elektroniką NeoGadżet.\n"
            "Użyj poniższych fragmentów bazy wiedzy (kontekstu), aby precyzyjnie odpowiedzieć na pytanie.\n"
            "Jeśli w kontekście nie ma odpowiedzi, powiedz uprzejmie, że nie posiadasz takich informacji i zachęć do kontaktu na pomoc@neogadzet.example.\n"
            "Odpowiadaj naturalnie, zwięźle, profesjonalnie i wyłącznie po polsku.\n\n"
            f"KONTEKST Z BAZY WIEDZY:\n{context}"
        )

        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.2
        )

        return {"response": chat_completion.choices[0].message.content}

    except Exception as e:
        print(f"Błąd podczas przetwarzania pytania: {e}")
        return {"response": "Przepraszam, wystąpił problem techniczny podczas generowania odpowiedzi. Spróbuj ponownie za chwilę."}

# Domyślny punkt wejścia dla wdrożenia produkcyjnego na Vercel (POST /api/chat)


@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    return await execute_chat_logic(request.message)

# Alias ułatwiający lokalne testowanie bezpośrednio z lokalnego pliku HTML (POST /chat)


@app.post("/chat")
async def chat_local(request: ChatRequest):
    return await execute_chat_logic(request.message)

# Przyjazna strona główna eliminująca błąd 404 podczas lokalnych testów (GET /)


@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "Serwer NeoAsystenta działa poprawnie! Aby rozmawiać z botem, otwórz plik index.html w przeglądarce.",
        "endpoints": {
            "chat_endpoint_vercel": "POST /api/chat",
            "chat_endpoint_local": "POST /chat",
            "health_check": "GET /api/health"
        }
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "openai_key_configured": OPENAI_API_KEY is not None,
        "is_cache_warm": cached_chunks_with_embeddings is not None
    }
