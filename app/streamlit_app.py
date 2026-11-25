"""
app/streamlit_app.py
--------------------
UI Streamlit : RAG classique | Graph RAG | Agentic RAG.

Lance depuis la RACINE :
    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path : app/ → racine
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv
from src.domain_guard import OUT_OF_DOMAIN_MESSAGE, is_fashion_question

load_dotenv(ROOT / ".env")


# =========================================================================
# CSS
# =========================================================================
st.set_page_config(page_title="DressMe — AI Stylist", page_icon="👗", layout="wide")

st.markdown("""
<style>
.mode-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.badge-classic { background:#2d4a3e; color:#7ed4a7; border:1px solid #3d6a5e; }
.badge-graph   { background:#2d3a5a; color:#7ab4f0; border:1px solid #3d5a8a; }
.badge-agentic { background:#4a2d4a; color:#d07af0; border:1px solid #6a3d7a; }

.step-item {
    padding: 5px 11px;
    margin: 3px 0;
    border-radius: 7px;
    font-size: 13px;
    background: #1e1e2a;
    border-left: 3px solid #6a3d7a;
    color: #c8b8d8;
}
.step-done { border-left-color: #7ed4a7; color: #7ed4a7; }

.agent-card {
    background: #1a2030;
    border: 1px solid #7ab4f0;
    border-radius: 10px;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 13px;
    color: #c8d8f0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Loaders cachés
# =========================================================================

@st.cache_resource(show_spinner="Initialisation RAG classique…")
def get_classic_rag():
    try:
        from src.rag.rag_engine import DressMeRAG
        rag = DressMeRAG()
        rag.initialize(force_rebuild=False)
        return rag
    except Exception as e:
        st.warning(f"RAG classique non chargé. Détail : {e}")
        return None


@st.cache_resource(show_spinner="Connexion au graphe…")
def get_graph_query():
    try:
        from src.graph.graph_query import GraphQuery
        return GraphQuery()
    except Exception:
        return None


@st.cache_resource(show_spinner="Initialisation Graph RAG…")
def get_graph_rag():
    from src.graph.graph_rag import GraphRAG
    return GraphRAG()


@st.cache_resource(show_spinner="Initialisation Agentic RAG…")
def get_agentic_rag():
    import os
    from src.agent.agentic_rag import AgenticRAG
    rag = get_classic_rag()
    gq  = get_graph_query()
    return AgenticRAG(
        rag_engine=rag,
        graph_query=gq,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )


# =========================================================================
# Layout — titre
# =========================================================================

st.title("👗 DressMe — AI Stylist")
st.caption("RAG classique · Graph RAG · Agentic RAG  •  Neo4j + ChromaDB + Groq")


# =========================================================================
# Sidebar
# =========================================================================

with st.sidebar:
    st.header("⚙️ Mode")
    mode = st.radio(
        "Choisis le moteur :",
        options=["Graph RAG", "RAG classique", "Agentic RAG 🤖"],
        index=0,
    )

    st.markdown("---")

    # ── Explorer le graphe (inchangé) ──────────────────────────────────────
    st.header("🔍 Explorer le graphe")
    try:
        gq = get_graph_query()
        if gq is None:
            raise ConnectionError("Neo4j inaccessible")
        styles = gq.list_styles()
        if styles:
            chosen = st.selectbox("Voir un style :", ["—"] + styles)
            if chosen and chosen != "—":
                profile = gq.style_full_profile(chosen)
                st.markdown(f"### {profile['style']}")
                st.markdown(f"**Vêtements** · {', '.join(profile['clothing']) or '—'}")
                st.markdown(f"**Couleurs** · {', '.join(profile['colors']) or '—'}")
                st.markdown(f"**Occasions** · {', '.join(profile['occasions']) or '—'}")
                st.markdown(f"**Saisons** · {', '.join(profile['seasons']) or '—'}")
                st.markdown(f"**Marques** · {', '.join(profile['brands']) or '—'}")
                if profile.get("inspired_by"):
                    st.markdown(f"**Inspiré par** · {', '.join(profile['inspired_by'])}")
        else:
            st.info("Graphe vide. Lance `python scripts/build_graph.py`.")
    except Exception as e:
        st.error(f"Neo4j inaccessible : {e}")

    st.markdown("---")

    # ── Mémoire + Exemples (Agentic uniquement) ────────────────────────────
    if mode == "Agentic RAG 🤖":
        st.header("🧠 Mémoire session")
        try:
            agent = get_agentic_rag()
            mem = agent.memory.summary()
            st.json(mem)
            if st.button("🗑️ Effacer mémoire"):
                agent.clear_memory()
                st.success("Mémoire effacée !")
        except Exception:
            st.info("Agentic RAG non initialisé.")

        st.markdown("---")
        st.header("💡 Exemples")
        examples = [
            "Tenue élégante mariage hiver Paris",
            "Compare Old Money et Minimalist pour entretien",
            "Outfit streetwear inspiré tendances japonaises",
            "Construis une tenue complète avec accessoires",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:18]}", use_container_width=True):
                st.session_state["_pending"] = ex

    st.markdown("---")
    st.caption("💡 Pose la même question dans les 3 modes pour voir la différence.")

    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================================
# État de session
# =========================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = mode

if st.session_state.current_mode != mode:
    st.session_state.messages = []
    st.session_state.current_mode = mode


# =========================================================================
# Helper rendu workflow agentique — défini AVANT la boucle historique
# =========================================================================
def _render_agent_data(ad: dict, inside_expander: bool = False) -> None:
    """Affiche métriques, plan, agents, trace, faits, passages, critic.
    
    inside_expander=True : contexte historique — pas de st.expander imbriqués.
    """
    plan        = ad.get("plan")
    steps_trace = ad.get("steps_trace", [])
    agents_used = ad.get("agents_used", [])
    critic      = ad.get("critic")
    graph_facts = ad.get("graph_facts", [])
    passages    = ad.get("passages", [])
    scores      = ad.get("scores", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", len(agents_used))
    c2.metric("Étapes", len(steps_trace))
    c3.metric("Complexité", plan.complexity if plan else "?")
    c4.metric("Cohérence", f"{critic.coherence_score:.0%}" if critic else "N/A")

    col_plan, col_agents = st.columns(2)
    with col_plan:
        st.markdown("**📋 Plan d'exécution**")
        if plan:
            st.markdown(f"*Intent : `{plan.intent}`*")
            for step in plan.steps:
                st.markdown(
                    f'<div class="step-item step-done">✓ {step}</div>',
                    unsafe_allow_html=True,
                )
    with col_agents:
        st.markdown("**🤖 Agents actifs**")
        icons = {
            "PlannerAgent": "🧠", "VectorAgent": "🔍",
            "GraphAgent": "🕸️", "OutfitAgent": "👗", "CriticAgent": "🔎",
        }
        for ag in agents_used:
            st.markdown(
                f'<div class="agent-card">{icons.get(ag,"🤖")} <b>{ag}</b></div>',
                unsafe_allow_html=True,
            )

    def _section(label: str, content_fn):
        """Expander en mode live, titre markdown + contenu direct en mode historique."""
        if inside_expander:
            st.markdown(f"**{label}**")
            content_fn()
        else:
            with st.expander(label):
                content_fn()

    def _trace_content():
        for step in steps_trace:
            st.markdown(f'<div class="step-item">{step}</div>', unsafe_allow_html=True)

    def _facts_content():
        for fact in graph_facts:
            st.code(fact, language="")

    def _passages_content():
        for i, (p, sc) in enumerate(zip(passages, scores or [0]*len(passages)), 1):
            st.markdown(f"**Passage {i}** — pertinence `{sc:.2f}`")
            st.markdown(f"> {p[:300]}…")

    def _critic_content():
        st.markdown(f"Score cohérence : **{critic.coherence_score:.0%}**")
        st.markdown(f"Validé : {'✅ Oui' if critic.validated else '❌ Non'}")
        st.markdown(f"Note : {critic.final_note}")
        for iss in (critic.issues or []):
            st.markdown(f"- ⚠️ {iss}")
        for imp in (critic.improvements or []):
            st.markdown(f"- 💡 {imp}")

    if steps_trace:
        _section("📜 Trace complète", _trace_content)
    if graph_facts:
        _section("🕸️ Faits du Knowledge Graph", _facts_content)
    if passages:
        _section("📄 Passages ChromaDB", _passages_content)
    if critic:
        _section("🔎 Rapport Critic Agent", _critic_content)
# =========================================================================
# Affichage historique
# =========================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        extras     = msg.get("extras") or {}
        agent_data = msg.get("agent_data")

        if extras:
            with st.expander("🔎 Voir le contexte utilisé"):
                if extras.get("keywords"):
                    st.markdown(f"**Mots-clés détectés** · `{extras['keywords']}`")
                if extras.get("graph_facts"):
                    st.markdown("**Faits du graphe :**")
                    for f in extras["graph_facts"]:
                        st.markdown(f"- {f}")
                if extras.get("vector_passages"):
                    st.markdown("**Passages vectoriels :**")
                    for i, p in enumerate(extras["vector_passages"], 1):
                        st.markdown(f"> *[{i}]* {p[:300]}…")
                if extras.get("sources"):
                    st.markdown("**Sources :**")
                    for s in extras["sources"]:
                        st.markdown(f"- `{s}`")

        if agent_data:
            with st.expander("🤖 Voir le workflow agentique"):
                _render_agent_data(agent_data, inside_expander=True)


# =========================================================================
# Input utilisateur
# =========================================================================

pending  = st.session_state.pop("_pending", None)
question = st.chat_input("Pose ta question mode…") or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Réflexion ({mode})…"):

            # ── FILTRE MÉTIER COMMUN ──────────────────────────────────────
            if not is_fashion_question(question):
                answer_text = OUT_OF_DOMAIN_MESSAGE
                st.warning(answer_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text}
                )

            # ── RAG CLASSIQUE ─────────────────────────────────────────────
            elif mode == "RAG classique":
                st.markdown(
                    '<span class="mode-badge badge-classic">Classic RAG</span>',
                    unsafe_allow_html=True,
                )
                rag    = get_classic_rag()
                extras = {}
                if rag is None:
                    answer_text = "RAG classique indisponible."
                else:
                    try:
                        out = rag.ask(question)
                        answer_text = out.get("answer", "(pas de réponse)")
                        extras = {"sources": out.get("sources", [])}
                    except Exception as e:
                        answer_text = f"Erreur RAG classique : {e}"

                st.markdown(answer_text)
                if extras:
                    with st.expander("🔎 Voir le contexte utilisé"):
                        if extras.get("sources"):
                            st.markdown("**Sources :**")
                            for s in extras["sources"]:
                                st.markdown(f"- `{s}`")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text, "extras": extras}
                )

            # ── GRAPH RAG ─────────────────────────────────────────────────
            elif mode == "Graph RAG":
                st.markdown(
                    '<span class="mode-badge badge-graph">Graph RAG</span>',
                    unsafe_allow_html=True,
                )
                grag   = get_graph_rag()
                result = grag.answer(question)
                answer_text = result.answer
                extras = {
                    "keywords":        result.keywords,
                    "graph_facts":     result.graph_facts,
                    "vector_passages": result.vector_passages,
                }

                st.markdown(answer_text)
                with st.expander("🔎 Voir le contexte utilisé"):
                    st.markdown(f"**Mots-clés détectés** · `{extras['keywords']}`")
                    if extras["graph_facts"]:
                        st.markdown("**Faits du graphe :**")
                        for f in extras["graph_facts"]:
                            st.markdown(f"- {f}")
                    else:
                        st.markdown("_Pas de fait pertinent dans le graphe._")
                    if extras["vector_passages"]:
                        st.markdown("**Passages vectoriels :**")
                        for i, p in enumerate(extras["vector_passages"], 1):
                            st.markdown(f"> *[{i}]* {p[:300]}…")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text, "extras": extras}
                )

            # ── AGENTIC RAG ───────────────────────────────────────────────
            elif mode == "Agentic RAG 🤖":
                st.markdown(
                    '<span class="mode-badge badge-agentic">Agentic RAG</span>',
                    unsafe_allow_html=True,
                )
                agent  = get_agentic_rag()
                result = agent.ask(question)

                st.markdown(result.answer)
                st.markdown("---")

                agent_data = {
                    "plan":        result.plan,
                    "steps_trace": result.steps_trace,
                    "agents_used": result.agents_used,
                    "critic":      result.critic_result,
                    "graph_facts": result.graph_result.facts
                                   if result.graph_result and result.graph_result.available
                                   else [],
                    "passages":    result.vector_result.passages
                                   if result.vector_result else [],
                    "scores":      result.vector_result.scores
                                   if result.vector_result else [],
                }
                _render_agent_data(agent_data)

                with st.expander("🧠 Mémoire mise à jour"):
                    st.json(result.memory_summary)

                st.session_state.messages.append(
                    {
                        "role":       "assistant",
                        "content":    result.answer,
                        "agent_data": agent_data,
                    }
                )