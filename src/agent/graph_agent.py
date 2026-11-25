"""
DressMe AI — Graph Retrieval Agent
Interroge Neo4j via GraphQuery pour extraire les faits relationnels mode.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GraphResult:
    facts: list[str]
    styles_found: list[str]
    relations_used: list[str]
    raw_data: dict
    available: bool = True


class GraphAgent:
    """
    Wraps GraphQuery pour l'interrogation intelligente du knowledge graph.
    Adapte les requêtes selon le plan fourni par le PlannerAgent.
    """

    def __init__(self, graph_query=None):
        """
        graph_query : instance de GraphQuery (peut être None si Neo4j indisponible).
        """
        self.gq = graph_query
        self._available = graph_query is not None

    def retrieve(
        self,
        keywords: list[str],
        styles_target: list[str] = None,
        occasion: Optional[str] = None,
        season: Optional[str] = None,
        need_comparison: bool = False,
    ) -> GraphResult:
        """
        Récupère les faits graphe pertinents selon le contexte du plan.
        """
        if not self._available:
            return GraphResult(
                facts=["[Neo4j non disponible — mode dégradé]"],
                styles_found=[],
                relations_used=[],
                raw_data={},
                available=False,
            )

        facts = []
        styles_found = []
        relations_used = []
        raw_data = {}

        try:
            # 1. Styles explicitement nommés → profil complet
            if styles_target:
                for style in styles_target:
                    profile = self.gq.style_full_profile(style)
                    if profile:
                        raw_data[style] = profile
                        styles_found.append(style)
                        relations_used.append("STYLE_PROFILE")
                        facts.append(self._format_profile(style, profile))

            # 2. Recherche par keywords
            if keywords:
                kw_styles = self.gq.search_by_keywords(keywords)
                for item in kw_styles[:3]:
                    s_name = item.get("style", "")
                    if s_name and s_name not in styles_found:
                        profile = self.gq.style_full_profile(s_name)
                        if profile:
                            raw_data[s_name] = profile
                            styles_found.append(s_name)
                            relations_used.append("KEYWORD_SEARCH")
                            facts.append(self._format_profile(s_name, profile))

            # 3. Filtre par occasion
            if occasion:
                occ_styles = self.gq.styles_for_occasion(occasion)
                if occ_styles:
                    relations_used.append("FITS_OCCASION")
                    facts.append(
                        f"Styles adaptés à '{occasion}' : {', '.join(occ_styles[:5])}"
                    )

            # 4. Filtre par saison
            if season:
                sea_styles = self.gq.styles_for_season(season)
                if sea_styles:
                    relations_used.append("POPULAR_IN_SEASON")
                    facts.append(
                        f"Styles populaires en {season} : {', '.join(sea_styles[:5])}"
                    )

            # 5. Styles similaires (pour comparaison)
            if need_comparison and len(styles_found) >= 1:
                for style in styles_found[:2]:
                    similar = self.gq.similar_styles(style, min_shared=1)
                    if similar:
                        relations_used.append("SIMILAR_STYLES")
                        facts.append(
                            f"Styles proches de {style} : {', '.join(similar[:3])}"
                        )

        except Exception as e:
            facts.append(f"[Erreur graphe partielle: {e}]")

        return GraphResult(
            facts=facts,
            styles_found=styles_found,
            relations_used=list(set(relations_used)),
            raw_data=raw_data,
            available=True,
        )

    def _format_profile(self, style_name: str, profile: dict) -> str:
        """Formate un profil de style en texte structuré."""
        lines = [f"▸ Style {style_name.upper()} :"]

        if profile.get("clothing"):
            lines.append(f"  Vêtements : {', '.join(profile['clothing'][:6])}")
        if profile.get("colors"):
            lines.append(f"  Couleurs : {', '.join(profile['colors'][:5])}")
        if profile.get("occasions"):
            lines.append(f"  Occasions : {', '.join(profile['occasions'][:4])}")
        if profile.get("seasons"):
            lines.append(f"  Saisons : {', '.join(profile['seasons'])}")
        if profile.get("inspired_by"):
            lines.append(f"  Inspiré par : {', '.join(profile['inspired_by'][:3])}")

        return "\n".join(lines)

    def format_for_prompt(self, result: GraphResult) -> str:
        """Formate les faits graphe pour injection dans un prompt LLM."""
        if not result.available:
            return "Knowledge Graph non disponible."
        if not result.facts:
            return "Aucun fait graphe trouvé pour cette requête."
        return "\n\n".join(result.facts)