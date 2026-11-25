"""
DressMe AI — Memory Manager
Gère la mémoire conversationnelle et les préférences utilisateur.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class UserPreferences:
    styles: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    budget: Optional[str] = None
    avoid: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    query: str
    answer: str
    styles_mentioned: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryManager:
    """
    Mémoire conversationnelle légère — pas de dépendance externe.
    Stocke préférences + historique dans la session Streamlit.
    """

    def __init__(self, max_history: int = 6):
        self.max_history = max_history
        self.preferences = UserPreferences()
        self.history: list[ConversationTurn] = []
        self._style_vocab = {
            "streetwear", "old money", "minimalist", "bohemian",
            "sporty chic", "business formal", "casual", "elegant",
            "vintage", "preppy", "grunge", "athleisure"
        }
        self._color_vocab = {
            "noir", "blanc", "beige", "gris", "marine", "camel",
            "bordeaux", "vert", "bleu", "rouge", "rose", "jaune",
            "orange", "marron", "crème", "ivoire", "nude", "kaki"
        }

    # ── Extraction automatique des préférences ──────────────────────────────

    def extract_and_update(self, query: str, answer: str) -> None:
        """Analyse la query et met à jour les préférences automatiquement."""
        q_lower = query.lower()

        styles_found = [s for s in self._style_vocab if s in q_lower]
        colors_found = [c for c in self._color_vocab if c in q_lower]

        for s in styles_found:
            if s not in self.preferences.styles:
                self.preferences.styles.append(s)

        for c in colors_found:
            if c not in self.preferences.colors:
                self.preferences.colors.append(c)

        # Détection occasion
        occasion_keywords = {
            "mariage": "mariage", "travail": "professionnel",
            "entretien": "professionnel", "université": "casual",
            "soirée": "soirée", "sport": "sport", "plage": "casual"
        }
        for kw, occ in occasion_keywords.items():
            if kw in q_lower and occ not in self.preferences.occasions:
                self.preferences.occasions.append(occ)

        # Détection saison
        for season in ["hiver", "été", "printemps", "automne"]:
            if season in q_lower and season not in self.preferences.seasons:
                self.preferences.seasons.append(season)

        # Enregistrer le tour
        turn = ConversationTurn(
            query=query,
            answer=answer[:300],  # tronqué pour la mémoire
            styles_mentioned=styles_found,
        )
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ── Formatage pour injection dans les prompts ───────────────────────────

    def get_context_string(self) -> str:
        """Retourne un résumé mémoire à injecter dans les prompts."""
        parts = []

        if self.preferences.styles:
            parts.append(f"Styles préférés : {', '.join(self.preferences.styles)}")
        if self.preferences.colors:
            parts.append(f"Couleurs favorites : {', '.join(self.preferences.colors)}")
        if self.preferences.occasions:
            parts.append(f"Occasions fréquentes : {', '.join(self.preferences.occasions)}")
        if self.preferences.seasons:
            parts.append(f"Saisons mentionnées : {', '.join(self.preferences.seasons)}")
        if self.preferences.avoid:
            parts.append(f"À éviter : {', '.join(self.preferences.avoid)}")

        if self.history:
            last = self.history[-1]
            parts.append(f"Dernière question : \"{last.query[:80]}\"")

        return "\n".join(parts) if parts else "Aucune préférence enregistrée."

    def get_history_messages(self) -> list[dict]:
        """Format LangChain-compatible pour l'historique."""
        messages = []
        for turn in self.history[-4:]:  # 4 derniers tours max
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def clear(self) -> None:
        self.history.clear()
        self.preferences = UserPreferences()

    def summary(self) -> dict:
        return {
            "turns": len(self.history),
            "styles": self.preferences.styles,
            "colors": self.preferences.colors,
            "occasions": self.preferences.occasions,
        }