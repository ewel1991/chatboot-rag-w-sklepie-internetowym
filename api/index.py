import os
import sys
import math
import warnings
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicjalizacja bota (tylko jeśli klucz jest dostępny)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="NeoAsystent RAG API - Lekki i Odporny")

# Konfiguracja CORS (umożliwia bezpieczną komunikację z frontendu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalna pamięć podręczna na wektory bazy wiedzy (In-Memory Cache)
cached_chunks_with_embeddings = None

# Wbudowana baza wiedzy (zabezpieczenie fail-safe)
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
    """Oblicza podobieństwo cosinusowe (iloczyn skalarny znormalizowanych wektorów OpenAI)."""
    return dot_product(v1, v2)

def load_and_embed_knowledge_base():
    """Wczytuje bazę wiedzy i generuje dla niej embeddingi (wektory)."""
    global cached_chunks_with_embeddings
    
    if cached_chunks_with_embeddings is not None:
        return cached_chunks_with_embeddings

    if not client:
        raise ValueError("Klient OpenAI nie został zainicjalizowany. Brak klucza API OPENAI_API_KEY.")

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
                print(f"Pomyślnie wczytano bazę wiedzy: {path}")
                break
            except Exception as e:
                print(f"Błąd odczytu z {path}: {e}")

    if not content:
        print("Użycie wbudowanej bazy zapasowej (fail-safe).")
        content = DEFAULT_KNOWLEDGE_BASE

    separator = "===================================================================="
    if separator in content:
        raw_chunks = content.split(separator)
    else:
        raw_chunks = content.split("\n\n")

    chunks = [rc.strip() for rc in raw_chunks if rc.strip()]
    if not chunks:
        chunks = [content.strip()]

    # Generowanie wektorów dla bazy wiedzy
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )

    cached_chunks_with_embeddings = []
    for i, data in enumerate(response.data):
        cached_chunks_with_embeddings.append({
            "text": chunks[i],
            "embedding": data.embedding
        })

    return cached_chunks_with_embeddings

class ChatRequest(BaseModel):
    message: str

async def execute_chat_logic(message: str):
    """Logika obsługi zapytania czatu RAG."""
    if not OPENAI_API_KEY:
        return {"response": "Błąd: Brak klucza OPENAI_API_KEY w zmiennych środowiskowych Vercela. Przejdź do Settings -> Environment Variables, dodaj klucz, a następnie wykonaj Redeploy."}
        
    if not client:
        return {"response": "Błąd: Nie udało się zainicjalizować bota OpenAI. Sprawdź poprawność klucza OPENAI_API_KEY."}

    try:
        kb_data = load_and_embed_knowledge_base()
    except Exception as e:
        print(f"Błąd bazy wiedzy RAG: {e}")
        return {"response": f"Błąd bazy wiedzy RAG: {str(e)}. Upewnij się, że Twoje konto OpenAI posiada środki oraz poprawnie skonfigurowany klucz."}

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

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_chunks[:3]
        context = "\n\n---\n\n".join([text for _, text in top_k])

        # 3. Definicja roli i instrukcji dla modelu (Prompt)
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
        print(f"Błąd podczas przetwarzania zapytania OpenAI: {e}")
        return {"response": f"Błąd połączenia z OpenAI: {str(e)}. Upewnij się, że Twój klucz API jest poprawny oraz że nie przekroczyłeś limitów zużycia."}

# Jawne mapowanie wszystkich najpopularniejszych ścieżek
@app.post("/api/chat")
@app.post("/chat")
async def chat_api(request: ChatRequest):
    return await execute_chat_logic(request.message)

# Inteligentne koło ratunkowe (Catch-all) na wypadek nietypowego routingu Vercela
@app.post("/{path:path}")
async def catch_all_post(path: str, request: Request):
    print(f"Przechwycono dynamiczne zapytanie POST na ścieżkę: {path}")
    try:
        body = await request.json()
        message = body.get("message", "")
        if "chat" in path or "chat" in message or message:
            return await execute_chat_logic(message)
    except Exception as e:
        print(f"Błąd analizy catch-all: {e}")
    
    return {
        "error": f"Nieobsługiwany punkt końcowy: {path}",
        "suggested_endpoint": "POST /api/chat"
    }

@app.get("/")
async def root():
    return {
        "status": "active",
        "message": "Serwer bota działa! Aby czatować, otwórz plik /sklep/index.html",
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