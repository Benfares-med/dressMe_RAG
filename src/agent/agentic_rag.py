"""
DressMe AI — Agentic RAG Orchestrator
Orchestre le workflow multi-agents avec un StateGraph LangGraph.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Any
from typing_extensions import TypedDict

# LangGraph
from langgraph.graph import StateGraph, END

from src.domain_guard import OUT_OF_DOMAIN_MESSAGE, is_fashion_question
from .memory_manager import MemoryManager
from .planner_agent import PlannerAgent, ExecutionPlan
from .vector_agent import VectorAgent, VectorResult
from .graph_agent import GraphAgent, GraphResult
from .outfit_agent import OutfitAgent, OutfitResult, CriticAgent, CriticResult


# ═══════════════════════════════════════════════════════════════════════════
# STATE — partagé entre tous les nœuds du graphe
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # Entrée
    query: str
    memory_context: str

    # Plan
    plan: Optional[Any]           # ExecutionPlan

    # Résultats des agents
    vector_result: Optional[Any]  # VectorResult
    graph_result: Optional[Any]   # GraphResult
    outfit_result: Optional[Any]  # OutfitResult
    critic_result: Optional[Any]  # CriticResult

    # Trace d'exécution (pour affichage Streamlit)
    steps_trace: list[str]
    agents_used: list[str]
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════
# RÉSULTAT FINAL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgenticRAGResult:
    answer: str
    plan: Optional[ExecutionPlan]
    vector_result: Optional[VectorResult]
    graph_result: Optional[GraphResult]
    critic_result: Optional[CriticResult]
    steps_trace: list[str]
    agents_used: list[str]
    memory_summary: dict


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class AgenticRAG:
    """
    Workflow LangGraph multi-agents pour DressMe AI.

    Nœuds :
    1. plan        → PlannerAgent
    2. vector      → VectorAgent (conditionnel)
    3. graph       → GraphAgent (conditionnel)
    4. outfit      → OutfitAgent
    5. critic      → CriticAgent (conditionnel)
    6. end         → agrégation finale
    """

    def __init__(
        self,
        rag_engine,           # DressMeRAG existant
        graph_query=None,     # GraphQuery Neo4j (optionnel)
        groq_api_key: str = "",
        model: str = "llama-3.1-8b-instant",
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # Agents
        self.memory = MemoryManager()
        self.planner = PlannerAgent(api_key, model)
        self.vector_agent = VectorAgent(rag_engine)
        self.graph_agent = GraphAgent(graph_query)
        self.outfit_agent = OutfitAgent(api_key, model)
        self.critic_agent = CriticAgent(api_key, model)

        # Compiler le graphe LangGraph
        self.graph = self._build_graph()

    # ── Construction du StateGraph ──────────────────────────────────────────

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        # Nœuds — préfixés "node_" pour éviter conflit avec les clés de AgentState
        workflow.add_node("node_plan",   self._node_plan)
        workflow.add_node("node_vector", self._node_vector)
        workflow.add_node("node_graph",  self._node_graph)
        workflow.add_node("node_outfit", self._node_outfit)
        workflow.add_node("node_critic", self._node_critic)

        # Point d'entrée
        workflow.set_entry_point("node_plan")

        # Après plan → routing conditionnel
        workflow.add_conditional_edges(
            "node_plan",
            self._route_after_plan,
            {
                "vector_only": "node_vector",
                "graph_only":  "node_graph",
                "both":        "node_vector",   # vector d'abord, puis graph
                "direct":      "node_outfit",
            }
        )

        # Après vector → graph si nécessaire, sinon outfit
        workflow.add_conditional_edges(
            "node_vector",
            self._route_after_vector,
            {
                "graph":  "node_graph",
                "outfit": "node_outfit",
            }
        )

        # Après graph → toujours outfit
        workflow.add_edge("node_graph", "node_outfit")

        # Après outfit → critic si nécessaire, sinon END
        workflow.add_conditional_edges(
            "node_outfit",
            self._route_after_outfit,
            {
                "critic": "node_critic",
                "end":    END,
            }
        )

        # Après critic → END
        workflow.add_edge("node_critic", END)

        return workflow.compile()

    # ── Nœuds du graphe ────────────────────────────────────────────────────

    def _node_plan(self, state: AgentState) -> AgentState:
        trace = state.get("steps_trace", [])
        trace.append("🧠 Planner Agent — analyse de la requête")

        plan = self.planner.plan(
            query=state["query"],
            memory_context=state.get("memory_context", ""),
        )
        trace.append(f"📋 Plan : {plan.intent} | complexité : {plan.complexity}")
        trace.append(f"🔀 Sources : vector={plan.use_vector}, graph={plan.use_graph}")

        return {
            **state,
            "plan": plan,
            "steps_trace": trace,
            "agents_used": ["PlannerAgent"],
        }

    def _node_vector(self, state: AgentState) -> AgentState:
        plan: ExecutionPlan = state["plan"]
        trace = state["steps_trace"]
        agents = state.get("agents_used", [])

        trace.append("🔍 Vector Agent — recherche ChromaDB")
        result = self.vector_agent.retrieve(
            query=state["query"],
            keywords=plan.keywords,
            top_k=4,
        )
        trace.append(f"   → {len(result.passages)} passages récupérés")
        agents.append("VectorAgent")

        return {**state, "vector_result": result, "steps_trace": trace, "agents_used": agents}

    def _node_graph(self, state: AgentState) -> AgentState:
        plan: ExecutionPlan = state["plan"]
        trace = state["steps_trace"]
        agents = state.get("agents_used", [])

        trace.append("🕸️  Graph Agent — requête Neo4j")
        result = self.graph_agent.retrieve(
            keywords=plan.keywords,
            styles_target=plan.styles_target,
            occasion=plan.occasion,
            season=plan.season,
            need_comparison=plan.need_comparison,
        )
        status = f"{len(result.styles_found)} styles trouvés" if result.available else "Neo4j indisponible"
        trace.append(f"   → {status}")
        agents.append("GraphAgent")

        return {**state, "graph_result": result, "steps_trace": trace, "agents_used": agents}

    def _node_outfit(self, state: AgentState) -> AgentState:
        plan: ExecutionPlan = state["plan"]
        trace = state["steps_trace"]
        agents = state.get("agents_used", [])

        trace.append("👗 Outfit Agent — génération de la recommandation")

        # Formater les contextes
        vector_ctx = "Aucun contexte vectoriel."
        if state.get("vector_result"):
            vector_ctx = self.vector_agent.format_for_prompt(state["vector_result"])

        graph_ctx = "Aucun contexte graphe."
        if state.get("graph_result"):
            graph_ctx = self.graph_agent.format_for_prompt(state["graph_result"])

        result = self.outfit_agent.generate(
            query=state["query"],
            vector_context=vector_ctx,
            graph_context=graph_ctx,
            memory_context=state.get("memory_context", ""),
            plan_steps=plan.steps,
            need_comparison=plan.need_comparison,
            need_outfit=plan.need_outfit_build,
        )
        trace.append("   → Recommandation générée")
        agents.append("OutfitAgent")

        return {**state, "outfit_result": result, "steps_trace": trace, "agents_used": agents}

    def _node_critic(self, state: AgentState) -> AgentState:
        trace = state["steps_trace"]
        agents = state.get("agents_used", [])

        trace.append("🔎 Critic Agent — validation de la cohérence")
        outfit: OutfitResult = state["outfit_result"]

        result = self.critic_agent.validate(
            query=state["query"],
            recommendation=outfit.recommendation,
        )
        score = f"{result.coherence_score:.0%}"
        trace.append(f"   → Score cohérence : {score} | validé : {result.validated}")
        agents.append("CriticAgent")

        return {**state, "critic_result": result, "steps_trace": trace, "agents_used": agents}

    # ── Fonctions de routing ────────────────────────────────────────────────

    def _route_after_plan(self, state: AgentState) -> str:
        plan: ExecutionPlan = state["plan"]
        if plan.use_vector and plan.use_graph:
            return "both"
        elif plan.use_vector:
            return "vector_only"
        elif plan.use_graph:
            return "graph_only"
        return "direct"

    def _route_after_vector(self, state: AgentState) -> str:
        plan: ExecutionPlan = state["plan"]
        return "graph" if plan.use_graph else "outfit"

    def _route_after_outfit(self, state: AgentState) -> str:
        plan: ExecutionPlan = state["plan"]
        return "critic" if plan.need_critic else "end"

    # ── Point d'entrée public ───────────────────────────────────────────────

    def ask(self, query: str) -> AgenticRAGResult:
        """Lance le workflow multi-agents et retourne le résultat complet."""

        if not is_fashion_question(query):
            return AgenticRAGResult(
                answer=OUT_OF_DOMAIN_MESSAGE,
                plan=None,
                vector_result=None,
                graph_result=None,
                critic_result=None,
                steps_trace=["Question refusée par le filtre métier."],
                agents_used=[],
                memory_summary=self.memory.summary(),
            )

        memory_ctx = self.memory.get_context_string()

        initial_state: AgentState = {
            "query": query,
            "memory_context": memory_ctx,
            "plan": None,
            "vector_result": None,
            "graph_result": None,
            "outfit_result": None,
            "critic_result": None,
            "steps_trace": [],
            "agents_used": [],
            "error": None,
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            # Fallback si LangGraph échoue
            final_state = {**initial_state, "error": str(e)}
            final_state["steps_trace"].append(f"❌ Erreur workflow : {e}")

        # Récupération de la réponse finale
        outfit: OutfitResult = final_state.get("outfit_result")
        answer = outfit.recommendation if outfit else "Erreur dans le workflow agentique."

        # Mise à jour mémoire
        self.memory.extract_and_update(query, answer)

        return AgenticRAGResult(
            answer=answer,
            plan=final_state.get("plan"),
            vector_result=final_state.get("vector_result"),
            graph_result=final_state.get("graph_result"),
            critic_result=final_state.get("critic_result"),
            steps_trace=final_state.get("steps_trace", []),
            agents_used=final_state.get("agents_used", []),
            memory_summary=self.memory.summary(),
        )

    def clear_memory(self):
        self.memory.clear()