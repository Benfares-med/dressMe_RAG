"""
DressMe AI — Vector Retrieval Agent
Interroge ChromaDB et retourne les passages les plus pertinents.
"""

from dataclasses import dataclass, field


@dataclass
class VectorResult:
    passages: list[str]
    sources: list[str]
    scores: list[float]
    query_used: str


class VectorAgent:
    """
    Wraps le DressMeRAG existant pour l'extraction vectorielle seule.
    Compatible avec la classe DressMeRAG de rag_engine.py.
    """

    def __init__(self, rag_engine):
        """
        rag_engine : instance de DressMeRAG déjà initialisée.
        On réutilise son vectorstore directement.
        """
        self.rag = rag_engine
        self._vectorstore = None

    def _get_vectorstore(self):
        """Accès lazy au vectorstore ChromaDB."""
        if self._vectorstore is None:
            # DressMeRAG expose vectorstore via self.rag.vectorstore
            # ou via self.rag._vectorstore selon l'implémentation
            if hasattr(self.rag, "vectorstore"):
                self._vectorstore = self.rag.vectorstore
            elif hasattr(self.rag, "_vectorstore"):
                self._vectorstore = self.rag._vectorstore
        return self._vectorstore

    def retrieve(self, query: str, keywords: list[str] = None, top_k: int = 4) -> VectorResult:
        """
        Récupère les passages ChromaDB les plus pertinents.
        Enrichit la query avec les keywords si fournis.
        """
        # Enrichissement de la query
        enriched_query = query
        if keywords:
            extra = " ".join(k for k in keywords if k.lower() not in query.lower())
            if extra:
                enriched_query = f"{query} {extra}"

        try:
            vs = self._get_vectorstore()
            if vs is not None:
                # Utilisation directe du vectorstore
                docs = vs.similarity_search_with_score(enriched_query, k=top_k)
                passages = []
                sources = []
                scores = []
                for doc, score in docs:
                    passages.append(doc.page_content)
                    sources.append(doc.metadata.get("source", "inconnu"))
                    # ChromaDB retourne distance (0=parfait), on inverse
                    scores.append(round(1.0 - float(score), 3))

                return VectorResult(
                    passages=passages,
                    sources=sources,
                    scores=scores,
                    query_used=enriched_query,
                )
            else:
                # Fallback : utiliser ask() du RAG classique
                return self._fallback_retrieve(enriched_query)

        except Exception as e:
            return VectorResult(
                passages=[f"[Erreur vectorstore: {e}]"],
                sources=["error"],
                scores=[0.0],
                query_used=enriched_query,
            )

    def _fallback_retrieve(self, query: str) -> VectorResult:
        """Fallback si le vectorstore n'est pas directement accessible."""
        try:
            result = self.rag.ask(query)
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            sources = result.get("sources", []) if isinstance(result, dict) else []
            return VectorResult(
                passages=[answer],
                sources=[s.get("source", "rag") if isinstance(s, dict) else str(s) for s in sources[:4]],
                scores=[0.8],
                query_used=query,
            )
        except Exception as e:
            return VectorResult(
                passages=[f"[Fallback error: {e}]"],
                sources=["error"],
                scores=[0.0],
                query_used=query,
            )

    def format_for_prompt(self, result: VectorResult) -> str:
        """Formate les passages pour injection dans un prompt LLM."""
        if not result.passages:
            return "Aucun passage vectoriel trouvé."

        lines = []
        for i, (passage, score) in enumerate(zip(result.passages, result.scores), 1):
            lines.append(f"[Passage {i} — pertinence {score:.2f}]\n{passage[:400]}")
        return "\n\n".join(lines)