"""
build_graph.py
--------------
Script CLI à lancer une fois pour peupler Neo4j.

Usage :
    python scripts/build_graph.py             # ajoute seed + extrait TXT
    python scripts/build_graph.py --wipe      # vide d'abord
    python scripts/build_graph.py --no-llm    # seed uniquement (offline)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Permet d'exécuter depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.graph_builder import GraphBuilder  # noqa: E402

log = logging.getLogger("build_graph")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le graphe DressMe.")
    parser.add_argument("--wipe", action="store_true", help="Vide Neo4j avant.")
    parser.add_argument("--no-llm", action="store_true",
                        help="N'utilise pas le LLM (seed uniquement).")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data/styles"))
    args = parser.parse_args()

    with GraphBuilder() as gb:
        gb.setup_schema()

        if args.wipe:
            gb.wipe()
            gb.setup_schema()  # recréer les contraintes après DETACH DELETE

        gb.load_seed()

        if not args.no_llm:
            gb.load_from_directory(args.data_dir)
        else:
            log.info("LLM désactivé (--no-llm).")

        print("\n=== Stats finales ===")
        for k, v in gb.stats().items():
            print(f"  {k:12s} {v}")
        print()


if __name__ == "__main__":
    main()