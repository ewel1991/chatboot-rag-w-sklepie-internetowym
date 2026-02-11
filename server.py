import os
import sys
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

# --- DIAGNOSTYKA SYSTEMU ---
# Pozostawiamy logi startowe, abyś zawsze wiedziała, czy system ruszył poprawnie
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
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("BŁĄD: Brak klucza OPENAI_API_KEY w pliku .env!")
    sys.exit(1)

app = FastAPI(title="NeoAsystent RAG API")

# Konfiguracja CORS (umożliwia index.html komunikację z serwerem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None


def setup_rag():
    global rag_chain
    print("\n--- INICJALIZACJA SYSTEMU RAG (OpenAI) ---")

    try:
        # KROK 1: Ładowanie bazy wiedzy z pliku tekstowego
        file_path = "knowledge_base_for_RAG.txt"
        if not os.path.exists(file_path):
            print(f"BŁĄD: Nie znaleziono pliku {file_path}")
            return

        print(f"1. Wczytywanie pliku: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # KROK 4: Konfiguracja Modelu językowego (LLM)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
        print(f"BŁĄD PODCZAS STARTU: {e}")


# Uruchomienie inicjalizacji przy starcie aplikacji
setup_rag()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(
            status_code=503, detail="System RAG nie jest gotowy.")
    try:
        # Wywołanie łańcucha RAG (z kluczem 'input')
        result = rag_chain.invoke({"input": request.message})
        return {"response": result["answer"]}
    except Exception as e:
        print(f"Błąd API: {e}")
        return {"response": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."}

if __name__ == "__main__":
    import uvicorn
    # Start serwera na localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
