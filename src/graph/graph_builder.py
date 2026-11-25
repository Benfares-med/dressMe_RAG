"""
graph_builder.py
----------------
Construit le graphe Neo4j à partir des descriptions de styles.

Adaptations à la structure DressMe :
  - Lit depuis  ./data/documents/*.txt  (comme rag_engine.py)
  - Un fichier peut contenir PLUSIEURS styles, séparés par "STYLE:"
  - Pour chaque bloc → 1 appel LLM → 1 upsert dans Neo4j

Stratégie hybride :
  1) Seed hardcodé (6 styles classiques, démo offline OK)
  2) Extraction LLM par bloc STYLE:
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from neo4j import GraphDatabase, Driver

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graph_builder")

load_dotenv()

# Chemins alignés sur rag_engine.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "documents"


# ===========================================================================
# 1. SEED
# ===========================================================================
SEED: dict[str, dict[str, Any]] = {
    "Streetwear": {
        "clothing": ["Oversized Hoodie", "Cargo Pants", "Sneakers", "Graphic Tee"],
        "colors": ["Black", "White", "Grey", "Neon Green"],
        "occasions": ["Casual", "Weekend", "Concert"],
        "seasons": ["Spring", "Autumn", "Winter"],
        "brands": ["Supreme", "Off-White", "Stüssy"],
        "inspired_by": ["Hip-Hop", "Skater"],
    },
    "Old Money": {
        "clothing": ["Blazer", "Polo Shirt", "Loafers", "Tailored Trousers"],
        "colors": ["Beige", "Navy", "Cream", "Forest Green"],
        "occasions": ["Brunch", "Country Club", "Wedding Guest"],
        "seasons": ["Spring", "Summer", "Autumn"],
        "brands": ["Ralph Lauren", "Brunello Cucinelli", "Loro Piana"],
        "inspired_by": ["Preppy", "Classic"],
    },
    "Minimalist": {
        "clothing": ["White Shirt", "Black Trousers", "Trench Coat", "Leather Belt"],
        "colors": ["White", "Black", "Grey", "Beige"],
        "occasions": ["Office", "Business Meeting", "Dinner"],
        "seasons": ["All Seasons"],
        "brands": ["COS", "Uniqlo", "The Row"],
        "inspired_by": ["Scandinavian", "Japanese"],
    },
    "Bohemian": {
        "clothing": ["Maxi Dress", "Embroidered Blouse", "Sandals", "Wide Hat"],
        "colors": ["Terracotta", "Mustard", "Cream", "Olive"],
        "occasions": ["Festival", "Beach", "Vacation"],
        "seasons": ["Spring", "Summer"],
        "brands": ["Free People", "Spell", "Isabel Marant"],
        "inspired_by": ["Hippie", "Folk"],
    },
    "Sporty Chic": {
        "clothing": ["Track Jacket", "Leggings", "Sneakers", "Cap"],
        "colors": ["Black", "White", "Pastel Pink", "Grey"],
        "occasions": ["Gym", "Casual", "Travel"],
        "seasons": ["All Seasons"],
        "brands": ["Lululemon", "Nike", "Adidas"],
        "inspired_by": ["Athleisure"],
    },
    "Business Formal": {
        "clothing": ["Suit", "Dress Shirt", "Oxford Shoes", "Silk Tie"],
        "colors": ["Navy", "Charcoal", "White", "Black"],
        "occasions": ["Office", "Conference", "Job Interview"],
        "seasons": ["All Seasons"],
        "brands": ["Hugo Boss", "Brioni", "Tom Ford"],
        "inspired_by": ["Classic"],
    },
}


# ===========================================================================
# 2. EXTRACTION LLM
# ===========================================================================
EXTRACTION_PROMPT = """Tu es un expert en mode. Analyse la description ci-dessous
et extrais les entités/relations sous forme de JSON STRICT.

RÈGLES :
- Réponds UNIQUEMENT avec un objet JSON valide. Pas de texte autour, pas de markdown.
- Noms d'entités en anglais, Title Case (ex: "Oversized Hoodie", "Beige").
- Catégorie vide → liste vide [].
- Max 8 éléments par catégorie.

SCHÉMA :
{{
  "style": "Nom du style principal",
  "clothing": [...],
  "colors": [...],
  "occasions": [...],
  "seasons": ["Spring" | "Summer" | "Autumn" | "Winter" | "All Seasons"],
  "brands": [...],
  "inspired_by": [...]
}}

DESCRIPTION :
---
{text}
---

JSON :"""


def _safe_json_parse(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.warning(f"JSON invalide : {e}")
        return None


def split_style_blocks(text: str) -> list[str]:
    """
    Découpe un fichier TXT en blocs, un par 'STYLE:'.
    Gère les variations : 'STYLE:', 'STYLE :', début de fichier, etc.
    """
    # On split sur 'STYLE:' ou 'STYLE :' (avec ou sans espace, début ligne)
    parts = re.split(r"(?im)^\s*STYLE\s*:", text)
    blocks: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) > 30:  # filtre les fragments vides / trop courts
            # On remet le préfixe pour donner du contexte au LLM
            blocks.append(f"STYLE: {p}")
    # Si aucun marqueur trouvé, on traite tout le texte comme un seul bloc
    if not blocks and text.strip():
        blocks = [text.strip()]
    return blocks


# ===========================================================================
# 3. BUILDER
# ===========================================================================
class GraphBuilder:
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

        self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._verify()
        self._llm: ChatGroq | None = None

    def _verify(self) -> None:
        try:
            self.driver.verify_connectivity()
            log.info(f"Connecté à Neo4j @ {self.uri}")
        except Exception as e:
            log.error(f"Connexion Neo4j échouée : {e}")
            raise

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        return self._llm

    # ---- schéma ----
    def setup_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT style_name IF NOT EXISTS "
            "FOR (s:Style) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT clothing_name IF NOT EXISTS "
            "FOR (c:Clothing) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT color_name IF NOT EXISTS "
            "FOR (c:Color) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT occasion_name IF NOT EXISTS "
            "FOR (o:Occasion) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT season_name IF NOT EXISTS "
            "FOR (s:Season) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT brand_name IF NOT EXISTS "
            "FOR (b:Brand) REQUIRE b.name IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for c in constraints:
                session.run(c)
        log.info("Contraintes d'unicité créées.")

    def wipe(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        log.warning("Graphe vidé.")

    # ---- insertion ----
    def upsert_style(self, payload: dict[str, Any]) -> None:
        style_name = (payload.get("style") or "").strip()
        if not style_name:
            log.warning("Payload sans 'style', ignoré.")
            return

        cypher = """
        MERGE (s:Style {name: $style})
        WITH s
        UNWIND $clothing AS item
          MERGE (c:Clothing {name: item})
          MERGE (s)-[:USES]->(c)
        WITH s
        UNWIND $colors AS col
          MERGE (c:Color {name: col})
          MERGE (s)-[:HAS_COLOR]->(c)
        WITH s
        UNWIND $occasions AS occ
          MERGE (o:Occasion {name: occ})
          MERGE (s)-[:FITS]->(o)
        WITH s
        UNWIND $seasons AS seas
          MERGE (ss:Season {name: seas})
          MERGE (s)-[:POPULAR_IN]->(ss)
        WITH s
        UNWIND $brands AS br
          MERGE (b:Brand {name: br})
          MERGE (b)-[:WORKS_FOR]->(s)
        WITH s
        UNWIND $inspired_by AS insp
          MERGE (i:Style {name: insp})
          MERGE (s)-[:INSPIRED_BY]->(i)
        """
        params = {
            "style": style_name,
            "clothing": payload.get("clothing", []) or [],
            "colors": payload.get("colors", []) or [],
            "occasions": payload.get("occasions", []) or [],
            "seasons": payload.get("seasons", []) or [],
            "brands": payload.get("brands", []) or [],
            "inspired_by": payload.get("inspired_by", []) or [],
        }
        with self.driver.session(database=self.database) as session:
            session.run(cypher, params)
        log.info(f"  ✓ {style_name}")

    # ---- pipelines ----
    def load_seed(self) -> None:
        log.info("→ Chargement du seed…")
        for name, data in SEED.items():
            self.upsert_style({"style": name, **data})
        log.info(f"Seed terminé : {len(SEED)} styles.")

    def extract_from_text(self, text: str) -> dict | None:
        prompt = EXTRACTION_PROMPT.format(text=text[:4000])
        try:
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            return _safe_json_parse(raw)
        except Exception as e:
            log.error(f"Extraction LLM échouée : {e}")
            return None

    def load_from_directory(self, data_dir: str | Path) -> None:
        """
        Parcourt les .txt, découpe par 'STYLE:', extrait via LLM, insère.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            log.warning(f"Dossier introuvable : {data_dir}")
            return

        files = sorted(data_dir.glob("*.txt"))
        if not files:
            log.warning(f"Aucun .txt dans {data_dir}")
            return

        log.info(f"→ Extraction LLM depuis {len(files)} fichier(s)…")
        total_blocks = 0
        for fp in files:
            text = fp.read_text(encoding="utf-8").strip()
            if not text:
                continue
            blocks = split_style_blocks(text)
            log.info(f"  · {fp.name} → {len(blocks)} bloc(s)")
            for i, block in enumerate(blocks, 1):
                payload = self.extract_from_text(block)
                if payload and payload.get("style"):
                    self.upsert_style(payload)
                    total_blocks += 1
                else:
                    log.warning(f"    bloc {i} : extraction vide")
        log.info(f"Extraction terminée : {total_blocks} bloc(s) insérés.")

    # ---- diagnostic ----
    def stats(self) -> dict[str, int]:
        queries = {
            "Style": "MATCH (n:Style) RETURN count(n) AS c",
            "Clothing": "MATCH (n:Clothing) RETURN count(n) AS c",
            "Color": "MATCH (n:Color) RETURN count(n) AS c",
            "Occasion": "MATCH (n:Occasion) RETURN count(n) AS c",
            "Season": "MATCH (n:Season) RETURN count(n) AS c",
            "Brand": "MATCH (n:Brand) RETURN count(n) AS c",
            "Relations": "MATCH ()-[r]->() RETURN count(r) AS c",
        }
        out = {}
        with self.driver.session(database=self.database) as session:
            for label, q in queries.items():
                out[label] = session.run(q).single()["c"]
        return out


# ===========================================================================
if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))
    with GraphBuilder() as gb:
        gb.setup_schema()
        gb.load_seed()
        gb.load_from_directory(data_dir)
        log.info("--- Stats finales ---")
        for k, v in gb.stats().items():
            log.info(f"  {k}: {v}")