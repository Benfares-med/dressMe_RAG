"""
DressMe AI — RAG Engine (Niveau 1)
Utilise Groq (LLaMA 3) + ChromaDB + HuggingFace Embeddings
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Permet aussi l'exécution directe : python src/rag/rag_engine.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

from src.domain_guard import OUT_OF_DOMAIN_MESSAGE, is_fashion_question

load_dotenv()

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = PROJECT_ROOT
DOCS_DIR   = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# ── Prompt styliste ───────────────────────────────────────────────────────────
STYLIST_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es DressMe, un assistant styliste IA expert en mode.
Utilise uniquement les informations ci-dessous pour répondre.
Si tu ne sais pas, dis-le clairement plutôt qu'inventer.

Contexte mode :
{context}

Question : {question}

Réponds de façon structurée, utile et inspirante. 
Si on te demande une tenue, liste les pièces clairement.
"""
)


class DressMeRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.chain       = None

    # ── 1. Chargement et découpage des documents ──────────────────────────────
    def load_documents(self):
        docs = []
        for txt_file in DOCS_DIR.glob("*.txt"):
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
        print(f"[RAG] {len(docs)} document(s) chargé(s)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\nSTYLE:", "\n", " "]
        )
        chunks = splitter.split_documents(docs)
        print(f"[RAG] {len(chunks)} chunks créés")
        return chunks

    # ── 2. Création du vectorstore ────────────────────────────────────────────
    def build_vectorstore(self, chunks):
        print("[RAG] Chargement du modèle d'embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        print(f"[RAG] Vectorstore prêt — {self.vectorstore._collection.count()} vecteurs")

    # ── 3. Chargement depuis disque (si déjà indexé) ──────────────────────────
    def load_vectorstore(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings
        )
        print(f"[RAG] Vectorstore chargé — {self.vectorstore._collection.count()} vecteurs")

    # ── 4. Construction de la chaîne RAG ─────────────────────────────────────
    def build_chain(self):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": STYLIST_PROMPT},
            return_source_documents=True
        )
        print("[RAG] Chaîne RetrievalQA prête")

    # ── 5. Initialisation complète ────────────────────────────────────────────
    def initialize(self, force_rebuild=False):
        if CHROMA_DIR.exists() and not force_rebuild:
            print("[RAG] Chargement du vectorstore existant...")
            self.load_vectorstore()
        else:
            print("[RAG] Construction du vectorstore...")
            chunks = self.load_documents()
            self.build_vectorstore(chunks)
        self.build_chain()
        print("[RAG] ✅ Système prêt !")

    # ── 6. Question → Réponse ─────────────────────────────────────────────────
    def ask(self, question: str) -> dict:
        if not is_fashion_question(question):
            return {"answer": OUT_OF_DOMAIN_MESSAGE, "sources": []}
        if not self.chain:
            raise RuntimeError("RAG non initialisé. Appelle initialize() d'abord.")
        result = self.chain.invoke({"query": question})
        return {
            "answer": result.get("result", result.get("answer")),
            "sources": [doc.metadata.get("source", "?") for doc in result["source_documents"]]
        }


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = DressMeRAG()
    rag.initialize(force_rebuild=True)  # Forcer la reconstruction pour tester le pipeline complet

    questions = [
        "Qu'est-ce que le style Old Money ?",
        "Propose-moi une tenue streetwear pour aller à l'université.",
        "Quels styles conviennent pour un mariage ?"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = rag.ask(q)
        print(f"💬 {response['answer']}")
        print(f"📄 Sources : {response['sources']}")