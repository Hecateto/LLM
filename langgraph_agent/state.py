from typing import List, Optional
from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class Step(BaseModel):
    title: str = ""
    description: str = ""
    status: Literal["pending", "completed"] = "pending"

class Plan(BaseModel):
    goal: str = ""
    thought: str = ""
    steps: List[Step] = []

class State(MessagesState):
    user_message: str = ""
    plan: Plan = Plan()
    observations: List[BaseMessage] = []
    final_report: str = ""
    error: Optional[str] = None