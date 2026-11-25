"""
DressMe AI — Planner Agent
Analyse la requête utilisateur et génère un plan d'exécution structuré.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq


@dataclass
class ExecutionPlan:
    intent: str                          # intention principale détectée
    complexity: str                      # "simple" | "medium" | "complex"
    steps: list[str]                     # étapes ordonnées
    use_vector: bool                     # utiliser ChromaDB ?
    use_graph: bool                      # utiliser Neo4j ?
    need_comparison: bool                # comparer plusieurs styles ?
    need_outfit_build: bool              # construire outfit complet ?
    need_critic: bool                    # valider la réponse ?
    keywords: list[str]                  # mots-clés extraits
    styles_target: list[str]             # styles nommés explicitement
    occasion: Optional[str] = None
    season: Optional[str] = None
    reasoning: str = ""                  # pourquoi ce plan


SYSTEM_PROMPT = """Tu es le Planner Agent du système DressMe AI.
Ton rôle : analyser une question mode et retourner un plan JSON STRICT.

Retourne UNIQUEMENT ce JSON (pas de markdown, pas d'explication) :
{
  "intent": "recommendation|comparison|outfit_build|discovery|general",
  "complexity": "simple|medium|complex",
  "steps": ["étape 1", "étape 2", ...],
  "use_vector": true|false,
  "use_graph": true|false,
  "need_comparison": true|false,
  "need_outfit_build": true|false,
  "need_critic": true|false,
  "keywords": ["mot1", "mot2", ...],
  "styles_target": ["style1", "style2", ...],
  "occasion": "valeur ou null",
  "season": "valeur ou null",
  "reasoning": "explication courte du plan"
}

Règles de décision :
- use_vector=true si question sur ambiance/feeling/inspiration
- use_graph=true si question sur relations styles/occasions/couleurs spécifiques
- use_graph=true toujours si styles nommés explicitement
- need_comparison=true si "compare", "différence", "vs", "ou"
- need_outfit_build=true si "tenue complète", "outfit", "ensemble", "accessoires"
- need_critic=true si complexity="complex" ou need_comparison=true
- complexity="complex" si plusieurs contraintes simultanées"""


class PlannerAgent:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def plan(self, query: str, memory_context: str = "") -> ExecutionPlan:
        """Génère un plan d'exécution pour la query donnée."""

        user_msg = f"Question : {query}"
        if memory_context and memory_context != "Aucune préférence enregistrée.":
            user_msg += f"\n\nContexte utilisateur :\n{memory_context}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=500,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            # Nettoyer les backticks markdown si présents
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            return ExecutionPlan(**data)

        except (json.JSONDecodeError, TypeError, Exception):
            # Fallback heuristique si le LLM échoue
            return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> ExecutionPlan:
        """Plan de secours basé sur des règles simples."""
        q = query.lower()

        styles_known = [
            "streetwear", "old money", "minimalist", "bohemian",
            "sporty chic", "business formal"
        ]
        styles_found = [s for s in styles_known if s in q]

        need_comparison = any(w in q for w in ["compare", "vs", "différence", "ou"])
        need_outfit = any(w in q for w in ["tenue", "outfit", "ensemble", "accessoires"])
        has_occasion = any(w in q for w in ["mariage", "travail", "soirée", "université"])

        use_graph = bool(styles_found) or has_occasion
        use_vector = not use_graph or any(w in q for w in ["inspiration", "tendance", "ambiance"])

        complexity = "complex" if (need_comparison or need_outfit or len(styles_found) > 1) else \
                     "medium" if (use_graph and use_vector) else "simple"

        steps = ["Analyse de la question"]
        if use_vector:
            steps.append("Recherche vectorielle ChromaDB")
        if use_graph:
            steps.append("Requête graphe Neo4j")
        if need_comparison:
            steps.append("Comparaison des styles")
        if need_outfit:
            steps.append("Construction de la tenue")
        steps.append("Génération de la recommandation")
        if complexity == "complex":
            steps.append("Validation et critique")

        return ExecutionPlan(
            intent="comparison" if need_comparison else
                   "outfit_build" if need_outfit else "recommendation",
            complexity=complexity,
            steps=steps,
            use_vector=use_vector,
            use_graph=use_graph,
            need_comparison=need_comparison,
            need_outfit_build=need_outfit,
            need_critic=complexity == "complex",
            keywords=query.split()[:6],
            styles_target=styles_found,
            reasoning="Plan heuristique (fallback)",
        )