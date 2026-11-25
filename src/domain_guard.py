"""Filtre déterministe pour limiter DressMe aux questions liées à la mode."""

from __future__ import annotations

import re
import unicodedata


OUT_OF_DOMAIN_MESSAGE = (
    "Je suis DressMe, un assistant spécialisé en mode et en conseil vestimentaire. "
    "Je ne peux pas répondre à cette question hors de mon domaine."
)

FASHION_TERMS = {
    "accessoire",
    "accessoires",
    "blazer",
    "bohemian",
    "boheme",
    "business formal",
    "casual",
    "chaussure",
    "chaussures",
    "chemise",
    "collection",
    "couleur",
    "couleurs",
    "dress code",
    "elegant",
    "fashion",
    "habille",
    "habiller",
    "jupe",
    "look",
    "marque",
    "marques",
    "minimalist",
    "minimaliste",
    "mettre",
    "mode",
    "mocassin",
    "mocassins",
    "old money",
    "outfit",
    "pantalon",
    "porter",
    "robe",
    "sneakers",
    "streetwear",
    "style",
    "styles",
    "tenue",
    "tenues",
    "tendance",
    "tendances",
    "vetement",
    "vetements",
}

# Ces termes sont utiles pour les demandes courtes comme
# "une tenue pour un mariage en hiver" ou "et pour l'été ?".
FASHION_CONTEXT_TERMS = {
    "automne",
    "entretien",
    "ete",
    "festival",
    "hiver",
    "mariage",
    "plage",
    "printemps",
    "soiree",
    "travail",
    "universite",
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def is_fashion_question(question: str) -> bool:
    """Retourne True si la question contient une intention liée à la mode."""
    normalized = _normalize(question)
    if not normalized:
        return False

    terms = FASHION_TERMS | FASHION_CONTEXT_TERMS
    return any(_contains_term(normalized, term) for term in terms)