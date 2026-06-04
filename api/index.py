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

# Inicjalizacja klienta OpenAI (tylko jeśli klucz jest dostępny w środowisku)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="NeoAsystent RAG API - Lekki i Odporny")

# Konfiguracja CORS (umożliwia komunikację z Twoim lokalnym i zdalnym frontendem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalna pamięć podręczna na wektory bazy wiedzy (In-Memory Cache)
# Pozwala to zaoszczędzić limity i czas przy ponownych wywołaniach tej samej lambdy
cached_chunks_with_embeddings = None

# Wbudowana zapasowa baza wiedzy na wypadek problemów ze ścieżkami plików na Vercel (Fail-safe)
DEFAULT_KNOWLEDGE_BASE = """
Słuchawki AeroSound X2 kosztują 299 zł. Mają aktywną redukcję szumów (ANC), działają przez 8 godzin na jednym ładowaniu i obsługują ładowanie bezprzewodowe Qi.
HomeCam Mini 2K kosztuje 349 zł. Posiada rozdzielczość 2K, funkcję wykrywania ruchu oraz tryb nocny do monitorowania domu.
WiLink AX1800 kosztuje 449 zł. To nowoczesny router obsługujący standard Wi-Fi 6, wyposażony w technologię MU-MIMO.
FitTime Pro kosztuje 599 zł. Smartwatch posiada wbudowany GPS, pulsometr działający 24/7 oraz baterię trzymającą do 10 dni.
Darmowa dostawa w sklepie NeoGadżet obowiązuje dla wszystkich zamówień od kwoty 199 zł. Dla tańszych zamówień dostawa do paczkomatu kosztuje 12 zł, a kurierem 15 zł.
Zasady zwrotów: Każdy klient ma prawo do zwrotu zakupionego towaru w ciągu 14 dni bez podawania przyczyny. Koszt odesłania towaru pokrywa kupujący.
"""


def dot_product(v1, v2):
    """Oblicza iloczyn skalarny dwóch wektorów."""
    return sum(x * y for x, y in zip(v1, v2))


def cosine_similarity(v1, v2):
    """Oblicza podobieństwo cosinusowe dwóch wektorów."""
    # Ponieważ wektory z modelu text-embedding-3-small są już znormalizowane,
    # ich podobieństwo cosinusowe to po prostu iloczyn skalarny.
    return dot_product(v1, v2)


def load_and_embed_knowledge_base():
    """Wczytuje bazę wiedzy z pliku zewnętrznego lub uruchamia bazę wbudowaną."""
    global cached_chunks_with_embeddings

    if cached_chunks_with_embeddings is not None:
        return cached_chunks_with_embeddings

    if not client:
        raise ValueError(
            "Klient OpenAI nie jest zainicjalizowany. Brak klucza API.")

    # Próba odnalezienia pliku bazy wiedzy w różnych potencjalnych lokalizacjach na Vercel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "..", "knowledge_base_for_RAG.txt"),
        os.path.join(current_dir, "knowledge_base_for_RAG.txt"),
        "knowledge_base_for_RAG.txt",
        "./knowledge_base_for_RAG.txt",
        "/var/task/knowledge_base_for_RAG.txt"
    ]

    content = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"Pomyślnie wczytano bazę wiedzy z lokalizacji: {path}")
                break
            except Exception as e:
                print(f"Błąd odczytu z {path}: {e}")

    # Jeśli nie udało się wczytać pliku zewnętrznego, używamy wbudowanej bazy (fail-safe)
    if not content:
        print("Ostrzeżenie: Plik bazy wiedzy nie został odnaleziony. Uruchamianie wbudowanej bazy zapasowej.")
        content = DEFAULT_KNOWLEDGE_BASE

    # Podział tekstu według separatora
    separator = "===================================================================="
    if separator in content:
        raw_chunks = content.split(separator)
    else:
        raw_chunks = content.split("\n\n")

    chunks = []
    for rc in raw_chunks:
        cleaned = rc.strip()
        if cleaned:
            chunks.append(cleaned)

    if not chunks:
        chunks = [content.strip()]

    # Generowanie embeddingów dla wszystkich fragmentów (Batching)
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )

    # Łączenie oryginalnego tekstu z wygenerowanym wektorem
    cached_chunks_with_embeddings = []
    for i, data in enumerate(response.data):
        cached_chunks_with_embeddings.append({
            "text": chunks[i],
            "embedding": data.embedding
        })

    print(
        f"Pomyślnie przygotowano bazę RAG: {len(cached_chunks_with_embeddings)} fragmentów.")
    return cached_chunks_with_embeddings


class ChatRequest(BaseModel):
    message: str


async def execute_chat_logic(message: str):
    """Wspólna logika obsługi zapytania czatu RAG."""
    if not OPENAI_API_KEY or not client:
        return {"response": "Konfiguracja serwera niekompletna. Upewnij się, że dodałeś poprawny klucz OPENAI_API_KEY w panelu Vercel."}

    try:
        kb_data = load_and_embed_knowledge_base()
    except Exception as e:
        print(f"Błąd inicjalizacji bazy wiedzy RAG: {e}")
        raise HTTPException(
            status_code=500, detail="Nie udało się załadować bazy wiedzy RAG.")

    try:
        # 1. Generowanie wektora zapytania
        query_response = client.embeddings.create(
            input=message,
            model="text-embedding-3-small"
        )
        query_vector = query_response.data[0].embedding

        # 2. Wyszukiwanie semantyczne (Cosine Similarity)
        scored_chunks = []
        for item in kb_data:
            similarity = cosine_similarity(query_vector, item["embedding"])
            scored_chunks.append((similarity, item["text"]))

        # Sortowanie od najbardziej dopasowanego fragmentu
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Wybór 3 najlepszych fragmentów kontekstu
        top_k = scored_chunks[:3]
        context = "\n\n---\n\n".join([text for _, text in top_k])

        # 3. Prompt systemowy dla LLM
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
        print(f"Błąd podczas przetwarzania zapytania: {e}")
        return {"response": "Przepraszam, wystąpił problem techniczny podczas generowania odpowiedzi. Spróbuj ponownie za chwilę."}

# Domyślny punkt wejścia dla wdrożenia produkcyjnego na Vercel (POST /api/chat)


@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    return await execute_chat_logic(request.message)

# Alias ułatwiający lokalne testowanie bezpośrednio z lokalnego pliku HTML (POST /chat)


@app.post("/chat")
async def chat_local(request: ChatRequest):
    return await execute_chat_logic(request.message)

# Strona główna eliminująca błąd 404 podczas lokalnych testów (GET /)


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
