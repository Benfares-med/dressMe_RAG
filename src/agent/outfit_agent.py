"""
DressMe AI — Outfit Agent + Critic Agent
Génère des recommandations mode complètes et valide leur cohérence.
"""

import re
from dataclasses import dataclass
from groq import Groq


# ═══════════════════════════════════════════════════════════════════════════
# OUTFIT AGENT
# ═══════════════════════════════════════════════════════════════════════════

OUTFIT_SYSTEM_PROMPT = """Tu es DressMe AI, un styliste personnel expert.
Tu génères des recommandations mode précises, concrètes et inspirantes.

Directives :
- Sois spécifique dans les pièces proposées (ex: "blazer en laine gris anthracite")
- Toujours proposer une alternative budget-friendly
- Structure ta réponse clairement
- Tiens compte du contexte (saison, occasion, style demandé)
- Utilise les faits du graphe et les passages textuels fournis
- Rédige en français, ton chaud et professionnel
- Si demande de comparaison : compare clairement avec un tableau ou des points
- Si outfit complet : liste haut / bas / chaussures / accessoires"""


@dataclass
class OutfitResult:
    recommendation: str
    outfit_items: list[str]     # items détectés dans la réponse
    styles_covered: list[str]


class OutfitAgent:
    """
    Génère la recommandation finale en fusionnant contexte vectoriel + graphe.
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def generate(
        self,
        query: str,
        vector_context: str,
        graph_context: str,
        memory_context: str,
        plan_steps: list[str],
        need_comparison: bool = False,
        need_outfit: bool = False,
    ) -> OutfitResult:

        # Construction du prompt selon le type de demande
        task_instruction = ""
        if need_comparison:
            task_instruction = "\n⚡ TÂCHE : Fais une comparaison structurée et claire des styles mentionnés."
        elif need_outfit:
            task_instruction = "\n⚡ TÂCHE : Construis une tenue complète avec haut, bas, chaussures et accessoires. Propose une alternative budget."

        prompt = f"""QUESTION UTILISATEUR : {query}
{task_instruction}

═══ CONTEXTE MÉMOIRE ═══
{memory_context}

═══ FAITS DU GRAPHE (source structurée — priorité haute) ═══
{graph_context}

═══ PASSAGES DOCUMENTAIRES (source textuelle) ═══
{vector_context}

═══ PLAN EXÉCUTÉ ═══
{chr(10).join(f"✓ {s}" for s in plan_steps)}

Génère maintenant ta recommandation mode en te basant sur ces informations.
Privilégie les faits du graphe pour les données objectives (couleurs, pièces, occasions).
Utilise les passages pour l'ambiance, l'inspiration et les détails stylistiques."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": OUTFIT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=900,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
            return OutfitResult(
                recommendation=answer,
                outfit_items=self._extract_items(answer),
                styles_covered=self._extract_styles(answer),
            )
        except Exception as e:
            return OutfitResult(
                recommendation=f"Erreur de génération : {e}",
                outfit_items=[],
                styles_covered=[],
            )

    def _extract_items(self, text: str) -> list[str]:
        """Extrait les pièces vestimentaires mentionnées."""
        items_kw = [
            "blazer", "veste", "jean", "pantalon", "robe", "chemise",
            "t-shirt", "pull", "manteau", "sneakers", "boots", "mocassins",
            "sac", "ceinture", "écharpe", "casquette", "montre"
        ]
        found = [kw for kw in items_kw if kw in text.lower()]
        return found[:8]

    def _extract_styles(self, text: str) -> list[str]:
        """Extrait les styles mentionnés dans la réponse."""
        styles_kw = [
            "streetwear", "old money", "minimalist", "bohemian",
            "sporty chic", "business formal", "casual", "élégant"
        ]
        return [s for s in styles_kw if s in text.lower()]


# ═══════════════════════════════════════════════════════════════════════════
# CRITIC AGENT
# ═══════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM_PROMPT = """Tu es le Critic Agent de DressMe AI.
Tu analyses une recommandation mode et retournes une validation JSON STRICT.

Retourne UNIQUEMENT ce JSON :
{
  "is_coherent": true|false,
  "coherence_score": 0.0-1.0,
  "issues": ["problème 1", ...],
  "improvements": ["amélioration 1", ...],
  "validated": true|false,
  "final_note": "commentaire court"
}

Vérifie :
- Cohérence style/occasion/saison
- Présence de recommandations concrètes
- Absence de contradictions
- Qualité et utilité de la réponse"""


@dataclass
class CriticResult:
    is_coherent: bool
    coherence_score: float
    issues: list[str]
    improvements: list[str]
    validated: bool
    final_note: str


class CriticAgent:
    """
    Valide la cohérence de la recommandation produite par OutfitAgent.
    Utilisé seulement pour les requêtes complexes (évite la sur-ingénierie).
    """

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def validate(self, query: str, recommendation: str) -> CriticResult:
        prompt = f"""Question originale : {query}

Recommandation à valider :
{recommendation[:600]}

Analyse cette recommandation et retourne le JSON de validation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            import json
            data = json.loads(raw)
            return CriticResult(**data)
        except Exception:
            # Si le critic échoue, on valide par défaut
            return CriticResult(
                is_coherent=True,
                coherence_score=0.75,
                issues=[],
                improvements=["Validation automatique (fallback)"],
                validated=True,
                final_note="Réponse acceptée (critic indisponible)",
            )