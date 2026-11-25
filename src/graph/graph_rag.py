"""
graph_rag.py
------------
Moteur Graph RAG : fusionne retrieval VECTORIEL (ChromaDB) et retrieval
RELATIONNEL (Neo4j) avant d'interroger le LLM (Groq).

Pipeline pour chaque question utilisateur :

  1. Extraction de mots-clés depuis la question (heuristique légère + LLM).
  2. Retrieval vectoriel  → top-k passages sémantiquement proches.
  3. Retrieval graphe     → styles + relations qui matchent les mots-clés.
  4. Fusion en un contexte unique (texte libre + faits structurés).
  5. Génération via Groq avec prompt dédié.

Avantage vs RAG classique :
  - Le LLM voit explicitement les RELATIONS (Style → Occasion, Style → Color…),
    pas seulement des phrases isolées. Il raisonne mieux sur le "qui-va-avec-quoi".
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.domain_guard import OUT_OF_DOMAIN_MESSAGE, is_fashion_question
from src.graph.graph_query import GraphQuery

load_dotenv()
log = logging.getLogger("graph_rag")

# Chemins alignés sur rag_engine.py
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma_db"))
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ===========================================================================
# Résultat structuré
# ===========================================================================
@dataclass
class GraphRAGResult:
    answer: str
    vector_passages: list[str] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Extraction de mots-clés
# ===========================================================================
# Vocabulaire mode connu : suffit pour 90% des questions et coûte 0 token LLM.
# Si rien ne matche, on bascule sur un fallback LLM.
KNOWN_TERMS = {
    "occasions": [
        "office", "wedding", "interview", "party", "brunch", "beach",
        "casual", "weekend", "concert", "festival", "vacation", "gym",
        "dinner", "conference", "travel",
    ],
    "colors": [
        "black", "white", "beige", "navy", "grey", "cream", "olive",
        "terracotta", "mustard", "pastel", "neon", "charcoal", "forest",
    ],
    "seasons": ["spring", "summer", "autumn", "fall", "winter"],
    "styles": [
        "streetwear", "old money", "minimalist", "bohemian",
        "sporty chic", "athleisure", "business formal", "preppy",
    ],
}


def heuristic_keywords(question: str) -> list[str]:
    """Extraction rapide basée vocabulaire connu. Pas de LLM."""
    q = question.lower()
    found: list[str] = []
    for bucket in KNOWN_TERMS.values():
        for term in bucket:
            if term in q:
                found.append(term)
    # dé-doublonne en préservant l'ordre
    return list(dict.fromkeys(found))


# ===========================================================================
# Prompts
# ===========================================================================
SYSTEM_PROMPT = """Tu es DressMe, un styliste IA expert en mode.

Tu disposes de DEUX sources :
  1) PASSAGES — extraits textuels descriptifs (style, ambiance, conseils).
  2) FAITS DU GRAPHE — relations structurées vérifiées (Style→Vêtement,
     Style→Couleur, Style→Occasion, Style→Saison, Marque→Style…).

Règles :
  - Privilégie les FAITS DU GRAPHE quand il y a contradiction.
  - Cite explicitement les styles concernés.
  - Donne des recommandations concrètes (vêtements, couleurs, occasions).
  - Réponds en français, ton conversationnel mais précis.
  - Si l'info manque, dis-le honnêtement.
"""

USER_TEMPLATE = """QUESTION :
{question}

────────── PASSAGES (retrieval vectoriel) ──────────
{passages}

────────── FAITS DU GRAPHE (retrieval relationnel) ──────────
{graph_facts}

────────── RÉPONSE ──────────"""


# ===========================================================================
# Moteur
# ===========================================================================
class GraphRAG:
    """Orchestrateur Graph RAG."""

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        top_k: int = 4,
    ) -> None:
        self.top_k = top_k

        log.info("Chargement embeddings…")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        log.info(f"Chargement Chroma @ {chroma_dir}")
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embeddings,
        )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.4,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.graph = GraphQuery()

    def close(self) -> None:
        self.graph.close()

    # ------------------------------------------------------------------
    # Étapes
    # ------------------------------------------------------------------
    def _extract_keywords(self, question: str) -> list[str]:
        kws = heuristic_keywords(question)
        if kws:
            return kws

        # Fallback LLM : on demande 3 à 6 mots-clés
        prompt = (
            "Extrais 3 à 6 mots-clés MODE (vêtements, couleurs, occasions, "
            "saisons, styles) de cette question. Réponds en une liste séparée "
            "par des virgules, en minuscules, sans ponctuation.\n\n"
            f"Question : {question}\n\nMots-clés :"
        )
        try:
            raw = self.llm.invoke(prompt).content
            tokens = [t.strip().lower() for t in re.split(r"[,;\n]", raw) if t.strip()]
            return tokens[:6]
        except Exception as e:
            log.warning(f"Fallback LLM keywords échoué : {e}")
            return []

    def _vector_retrieve(self, question: str) -> list[str]:
        try:
            docs = self.vectorstore.similarity_search(question, k=self.top_k)
            return [d.page_content.strip() for d in docs]
        except Exception as e:
            log.warning(f"Retrieval vectoriel échoué : {e}")
            return []

    def _graph_retrieve(self, keywords: list[str]) -> list[str]:
        """Convertit les hits graphe en lignes textuelles compactes pour le LLM."""
        if not keywords:
            return []

        facts: list[str] = []
        hits = self.graph.search_by_keywords(keywords, limit=6)

        for hit in hits:
            style = hit["style"]
            profile = self.graph.style_full_profile(style)
            if not profile:
                continue
            line = (
                f"• {profile['style']} "
                f"| Vêtements: {', '.join(profile['clothing'][:5]) or '—'} "
                f"| Couleurs: {', '.join(profile['colors'][:5]) or '—'} "
                f"| Occasions: {', '.join(profile['occasions'][:5]) or '—'} "
                f"| Saisons: {', '.join(profile['seasons']) or '—'} "
                f"| Marques: {', '.join(profile['brands'][:3]) or '—'}"
            )
            facts.append(line)

        return facts

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def answer(self, question: str) -> GraphRAGResult:
        log.info(f"Q: {question}")

        if not is_fashion_question(question):
            return GraphRAGResult(answer=OUT_OF_DOMAIN_MESSAGE)

        keywords = self._extract_keywords(question)
        log.info(f"  keywords: {keywords}")

        passages = self._vector_retrieve(question)
        log.info(f"  vector passages: {len(passages)}")

        graph_facts = self._graph_retrieve(keywords)
        log.info(f"  graph facts: {len(graph_facts)}")

        # Fusion → prompt
        passages_txt = (
            "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
            or "(aucun passage pertinent)"
        )
        facts_txt = "\n".join(graph_facts) or "(aucun fait pertinent dans le graphe)"

        user_msg = USER_TEMPLATE.format(
            question=question,
            passages=passages_txt,
            graph_facts=facts_txt,
        )

        try:
            resp = self.llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.error(f"LLM échec : {e}")
            answer = f"Erreur LLM : {e}"

        return GraphRAGResult(
            answer=answer,
            vector_passages=passages,
            graph_facts=graph_facts,
            keywords=keywords,
            debug={
                "n_passages": len(passages),
                "n_graph_facts": len(graph_facts),
            },
        )