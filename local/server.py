import os
import sys
import warnings
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INTELIGENTNY FALLBACK DLA SLOWAPI ---
# Jeśli biblioteka slowapi nie jest zainstalowana w requirements.txt,
# serwer nie wygeneruje błędu, tylko pominie limitowanie żądań.
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    has_limiter = True
    print("[INFO] Pomyślnie załadowano zabezpieczenie SlowAPI.")
except ImportError:
    print("[OSTRZEŻENIE] Brak modułu 'slowapi'. Limitowanie żądań będzie nieaktywne.")
    has_limiter = False

    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = DummyLimiter()

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

# --- DIAGNOSTYKA SYSTEMU ---
print("\n=== DIAGNOSTYKA SYSTEMU ===")
print(f"1. Ścieżka Pythona: {sys.executable}")
try:
    import langchain
    import langchain_core
    print(f"2. Lokalizacja LangChain: {langchain.__file__}")
    print(
        f"3. Wersja LangChain: {getattr(langchain, '__version__', 'Nieznana')}")
    print(f"4. Wersja LangChain Core: {langchain_core.__version__}")
except Exception as e:
    print(f"BŁĄD DIAGNOSTYKI: {e}")
print("===========================\n")

try:
    # Importy z paczek specjalistycznych
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate

    # Importy z głównej paczki 'langchain'
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

except ImportError as e:
    print(f"\n[BŁĄD KRYTYCZNY]: Nie można załadować modułu: {e}")
    print("\nROZWIĄZANIE: Upewnij się, że zainstalowałeś: pip install langchain langchain-openai langchain-community faiss-cpu")
    sys.exit(1)

# 1. Konfiguracja środowiska
load_dotenv_path = Path(__file__).parent / ".env"
if load_dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=load_dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("BŁĄD: Brak klucza OPENAI_API_KEY w pliku .env lub zmiennych środowiskowych Vercela!")
    # Nie przerywamy działania na Vercelu, aby ułatwić debugowanie w przeglądarce
    OPENAI_API_KEY = "dummy_key_to_prevent_crash_at_startup"

app = FastAPI(title="NeoAsystent RAG API")
if has_limiter:
    app.state.limiter = limiter

# Konfiguracja CORS (umożliwia index.html komunikację z serwerem)
app.add_middleware(
    CORSMiddleware,
    # Zezwalamy na połączenia z dowolnego źródła na produkcji
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None


def setup_rag():
    global rag_chain
    print("\n--- INICJALIZACJA SYSTEMU RAG (OpenAI) ---")

    try:
        # KROK 1: Ładowanie bazy wiedzy z pliku tekstowego
        KB_PATH = Path(__file__).parent / "knowledge_base_for_RAG.txt"
        if not KB_PATH.exists():
            KB_PATH = Path("knowledge_base_for_RAG.txt")

        if not KB_PATH.exists():
            raise FileNotFoundError(
                f"Nie odnaleziono bazy wiedzy pod ścieżką: {KB_PATH.absolute()}")

        print(f"1. Wczytywanie pliku: {KB_PATH}")
        loader = TextLoader(str(KB_PATH), encoding="utf-8")
        raw_docs = loader.load()

        # KROK 2: Dzielenie tekstu na fragmenty (Chunks)
        print("2. Dzielenie tekstu na fragmenty...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "====================================================================", "\n\n", "\n", " "]
        )
        docs = text_splitter.split_documents(raw_docs)
        print(f"   - Utworzono {len(docs)} fragmentów.")

        # KROK 3: Wektoryzacja (Embeddingi)
        print("3. Tworzenie bazy wektorowej (Model: text-embedding-3-small)...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # KROK 4: Konfiguracja Modelu językowego (LLM)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                         openai_api_key=OPENAI_API_KEY)

        # KROK 5: Definicja instrukcji (Prompt)
        system_prompt = (
            "Jesteś NeoAsystentem, doradcą klienta w sklepie NeoGadżet. "
            "Używaj poniższych fragmentów bazy wiedzy, aby odpowiedzieć na pytanie. "
            "Jeśli nie znasz odpowiedzi, powiedz, że nie wiesz i poproś o kontakt na pomoc@neogadzet.example. "
            "Odpowiadaj uprzejmie i po polsku.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # KROK 6: Budowa łańcucha RAG
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        print("--- SYSTEM GOTOWY DO PRACY! ---")

    except Exception as e:
        print(f"BŁĄD PODCZAS STARTU RAG: {e}")


# Uruchomienie inicjalizacji przy starcie aplikacji
setup_rag()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

    @validator('message')
    def sanitize_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

# Dodanie obsługi obu ścieżek: z prefiksem /api (wymóg Vercela) oraz bez


@app.post("/api/chat")
@app.post("/chat")
# Max 10 requestów/min (jeśli slowapi jest dostępne)
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    if OPENAI_API_KEY == "dummy_key_to_prevent_crash_at_startup":
        return {"response": "Błąd: Brak klucza OPENAI_API_KEY w konfiguracji środowiskowej Vercela. Upewnij się, że dodałeś poprawny klucz w zakładce Settings -> Environment Variables w panelu Vercel i wykonałeś Redeploy."}

    if not rag_chain:
        raise HTTPException(
            status_code=503, detail="System RAG nie jest gotowy do pracy."
        )
    try:
        # Wywołanie łańcucha RAG (z kluczem 'input')
        result = rag_chain.invoke({"input": request.message})
        return {"response": result["answer"]}
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail="Nieprawidłowe zapytanie")
    except Exception as e:
        logger.error(f"RAG chain error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Błąd wewnętrzny przetwarzania RAG")

if __name__ == "__main__":
    import uvicorn
    # Start serwera na localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
