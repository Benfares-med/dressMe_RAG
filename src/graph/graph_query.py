"""
graph_query.py
--------------
Couche d'accès lecture au graphe Neo4j.

Chaque méthode encapsule une intention métier ("styles pour une occasion",
"vêtements d'un style", "styles partageant une couleur", etc.) et renvoie
des dicts Python prêts à être sérialisés pour le LLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()
log = logging.getLogger("graph_query")


class GraphQuery:
    """Interface de lecture sur le graphe mode."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "dressme123")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver: Driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Helper générique
    # ------------------------------------------------------------------
    def _run(self, cypher: str, **params) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [r.data() for r in session.run(cypher, **params)]

    # ==================================================================
    # Requêtes métier
    # ==================================================================

    def list_styles(self) -> list[str]:
        rows = self._run("MATCH (s:Style) RETURN s.name AS name ORDER BY name")
        return [r["name"] for r in rows]

    def style_full_profile(self, style: str) -> dict[str, Any]:
        """
        Retourne le profil complet d'un style :
        vêtements, couleurs, occasions, saisons, marques, inspirations.
        """
        cypher = """
        MATCH (s:Style)
        WHERE toLower(s.name) = toLower($style)
        OPTIONAL MATCH (s)-[:USES]->(c:Clothing)
        OPTIONAL MATCH (s)-[:HAS_COLOR]->(col:Color)
        OPTIONAL MATCH (s)-[:FITS]->(o:Occasion)
        OPTIONAL MATCH (s)-[:POPULAR_IN]->(se:Season)
        OPTIONAL MATCH (b:Brand)-[:WORKS_FOR]->(s)
        OPTIONAL MATCH (s)-[:INSPIRED_BY]->(i:Style)
        RETURN s.name AS style,
               collect(DISTINCT c.name)   AS clothing,
               collect(DISTINCT col.name) AS colors,
               collect(DISTINCT o.name)   AS occasions,
               collect(DISTINCT se.name)  AS seasons,
               collect(DISTINCT b.name)   AS brands,
               collect(DISTINCT i.name)   AS inspired_by
        """
        rows = self._run(cypher, style=style)
        return rows[0] if rows else {}

    def styles_for_occasion(self, occasion: str) -> list[dict]:
        cypher = """
        MATCH (s:Style)-[:FITS]->(o:Occasion)
        WHERE toLower(o.name) CONTAINS toLower($occasion)
        OPTIONAL MATCH (s)-[:HAS_COLOR]->(col:Color)
        OPTIONAL MATCH (s)-[:USES]->(c:Clothing)
        RETURN s.name AS style,
               collect(DISTINCT col.name)[..5] AS colors,
               collect(DISTINCT c.name)[..5]   AS key_items
        ORDER BY style
        """
        return self._run(cypher, occasion=occasion)

    def styles_with_colors(self, colors: list[str]) -> list[dict]:
        """Styles qui utilisent AU MOINS UNE des couleurs données."""
        cypher = """
        MATCH (s:Style)-[:HAS_COLOR]->(col:Color)
        WHERE any(c IN $colors WHERE toLower(col.name) = toLower(c))
        WITH s, collect(DISTINCT col.name) AS matched
        OPTIONAL MATCH (s)-[:FITS]->(o:Occasion)
        RETURN s.name AS style,
               matched,
               collect(DISTINCT o.name)[..5] AS occasions
        ORDER BY size(matched) DESC, style
        """
        return self._run(cypher, colors=colors)

    def styles_for_season(self, season: str) -> list[dict]:
        cypher = """
        MATCH (s:Style)-[:POPULAR_IN]->(se:Season)
        WHERE toLower(se.name) = toLower($season)
           OR toLower(se.name) = 'all seasons'
        OPTIONAL MATCH (s)-[:USES]->(c:Clothing)
        RETURN s.name AS style,
               collect(DISTINCT c.name)[..5] AS key_items
        ORDER BY style
        """
        return self._run(cypher, season=season)

    def clothing_to_styles(self, clothing: str) -> list[dict]:
        """Quel(s) style(s) utilise(nt) ce vêtement ?"""
        cypher = """
        MATCH (c:Clothing)<-[:USES]-(s:Style)
        WHERE toLower(c.name) CONTAINS toLower($clothing)
        RETURN c.name AS item, collect(DISTINCT s.name) AS styles
        """
        return self._run(cypher, clothing=clothing)

    def similar_styles(self, style: str, min_shared: int = 2) -> list[dict]:
        """
        Styles "proches" : ils partagent au moins N entités
        (vêtement, couleur, occasion ou saison) avec celui demandé.
        """
        cypher = """
        MATCH (s1:Style {name: $style})-[]->(x)<-[]-(s2:Style)
        WHERE s1 <> s2
        WITH s2, count(DISTINCT x) AS shared,
             collect(DISTINCT x.name)[..6] AS via
        WHERE shared >= $min_shared
        RETURN s2.name AS style, shared, via
        ORDER BY shared DESC
        LIMIT 5
        """
        return self._run(cypher, style=style, min_shared=min_shared)

    # ==================================================================
    # Recherche "intelligente" — utilisée par Graph RAG
    # ==================================================================
    def search_by_keywords(self, keywords: list[str], limit: int = 8) -> list[dict]:
        """
        Cherche les styles dont le nom OU une entité liée (vêtement,
        couleur, occasion, saison, marque) matche un des mots-clés.
        Retour scoré : plus il y a de matchs, plus c'est haut.
        """
        if not keywords:
            return []

        cypher = """
        WITH [k IN $keywords | toLower(k)] AS kws
        MATCH (s:Style)
        OPTIONAL MATCH (s)-[]->(x)
        OPTIONAL MATCH (b:Brand)-[:WORKS_FOR]->(s)
        WITH s, kws,
             collect(DISTINCT toLower(x.name)) + collect(DISTINCT toLower(b.name))
               + [toLower(s.name)] AS bag
        WITH s, kws,
             [k IN kws WHERE any(v IN bag WHERE v CONTAINS k)] AS hits
        WHERE size(hits) > 0
        RETURN s.name AS style, hits, size(hits) AS score
        ORDER BY score DESC, style
        LIMIT $limit
        """
        return self._run(cypher, keywords=keywords, limit=limit)