"""DressMe AI — Agent module (Niveau 3 : Agentic RAG)"""

from .memory_manager import MemoryManager, UserPreferences
from .planner_agent import PlannerAgent, ExecutionPlan
from .vector_agent import VectorAgent, VectorResult
from .graph_agent import GraphAgent, GraphResult
from .outfit_agent import OutfitAgent, CriticAgent, OutfitResult, CriticResult
from .agentic_rag import AgenticRAG, AgenticRAGResult

__all__ = [
    "MemoryManager", "UserPreferences",
    "PlannerAgent", "ExecutionPlan",
    "VectorAgent", "VectorResult",
    "GraphAgent", "GraphResult",
    "OutfitAgent", "CriticAgent", "OutfitResult", "CriticResult",
    "AgenticRAG", "AgenticRAGResult",
]