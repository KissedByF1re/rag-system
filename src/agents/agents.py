from dataclasses import dataclass

from langgraph.pregel import Pregel

from agents.rag_assistant import rag_assistant
from schema import AgentInfo

DEFAULT_AGENT = "rag_assistant"


@dataclass
class Agent:
    description: str
    graph: Pregel


agents: dict[str, Agent] = {
    "rag_assistant": Agent(
        description="A rag assistant agent", graph=rag_assistant
    ),
}


def get_agent(agent_id: str) -> Pregel:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
