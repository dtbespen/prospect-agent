from typing import Annotated, List, Dict, Union, TypedDict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv
from tools import linkedin_tool, hunter_tool
from prompts import (
    LINKEDIN_ANALYSIS_PROMPT, 
    PRIORITY_ANALYSIS_PROMPT,
    get_example_linkedin_analysis,
    get_example_priority_analysis
)
from models import (
    UserBase, 
    LinkedInAnalysis, 
    SearchConfig, 
    AgentState, 
    LinkedInRawData, 
    EnrichedUser, 
    HunterResponse,
    PriorityAnalysis
)
from reducers import add_messages, add_users
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager
import logging
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Konfigurer logging enda strengere
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Deaktiver logging for flere moduler
logging.getLogger('langchain').setLevel(logging.ERROR)
logging.getLogger('langchain_core').setLevel(logging.ERROR)
logging.getLogger('langchain_openai').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)

# Configs
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0

# Wrap OpenAI client for better tracing
openai_client = wrap_openai(OpenAI())

# Definer state typer
MessageList = Annotated[List[BaseMessage], "messages"]
UserList = Annotated[List[Dict], "users"]

# Nåværende implementering
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "messages"]
    users: Annotated[List[Dict], "users"]
    config: SearchConfig

# 3. SCREENING NODE (FØRSTE STEG)
@traceable(run_type="chain", name="hunter_collection")
def collect_hunter_data(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Henter kontakter fra Hunter.io"""
    try:
        hunter_data = hunter_tool.invoke({
            "domain": state["config"]["domain"],
            "api_key": os.getenv("HUNTER_API_KEY")
        })
        
        users = [
            {
                "email": email["value"],
                "first_name": email.get("first_name"),
                "last_name": email.get("last_name"),
                "role": email.get("position"),
                "linkedin_url": email.get("linkedin"),
                "confidence": str(email.get("confidence", "")),
                "sources": ["hunter"]
            }
            for email in hunter_data.get("emails", [])
        ]
        
        return {
            "messages": [
                ToolMessage(
                    tool_call_id="hunter_success",
                    tool_name="hunter",
                    content=f"Hentet {len(users)} kontakter"
                )
            ],
            "users": users
        }
    except Exception as e:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id="hunter_error",
                    tool_name="hunter",
                    content=f"Feil: {str(e)}"
                )
            ],
            "users": []
        }

# 4. PRIORITERING NODE (ANDRE STEG)
@traceable(run_type="chain", name="prioritize_users")
def prioritize_users(state: AgentState, config: RunnableConfig) -> Dict[str, Union[MessageList, UserList]]:
    """Prioriterer brukere basert på deres egnethet for målrollen."""
    
    users_with_roles = [u for u in state["users"] if u.get("role")]
    
    if not users_with_roles:
        return {
            "messages": [HumanMessage(content="Ingen brukere med roller funnet")],
            "users": []
        }
    
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE
    ).with_structured_output(PriorityAnalysis)
    
    analysis = model.invoke(
        PRIORITY_ANALYSIS_PROMPT.format(
            role=state['config']['target_role'],
            users=json.dumps(users_with_roles, indent=2),
            max_results=state['config'].get('max_results', 5),
            model_schema=json.dumps(PriorityAnalysis.model_json_schema(), indent=2)
        )
    )
    
    # Lag dictionary av analysene med email som nøkkel
    analysis_by_email = {
        priority_user.email: priority_user 
        for priority_user in analysis.users
    }
    
    # Match brukere med deres analyse
    prioritized = []
    for user in users_with_roles:
        if user["email"] in analysis_by_email:
            priority_user = analysis_by_email[user["email"]]
            prioritized.append({
                **user,
                "priority_score": priority_user.score,
                "priority_reason": priority_user.reason,
                "sources": user["sources"] + ["prioritized"]
            })
    
    return {
        "messages": [HumanMessage(content=f"Prioriterte {len(prioritized)} brukere")],
        "users": prioritized
    }

@traceable(run_type="chain", name="get_linkedin_info")
def get_linkedin_info(state: AgentState, config: RunnableConfig) -> Dict[str, Union[MessageList, UserList]]:
    """Beriker prioriterte brukere med LinkedIn data."""
    
    prioritized_users = [u for u in state["users"] 
                        if "prioritized" in u.get("sources", [])
                        and u.get("linkedin_url")]
    
    if not prioritized_users:
        return {
            "messages": [HumanMessage(content="Ingen brukere å berike")],
            "users": state["users"]
        }
    
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE
    ).with_structured_output(LinkedInAnalysis)
    
    enriched = []
    messages = []
    
    for user in prioritized_users:
        try:
            linkedin_data = linkedin_tool.invoke({"linkedin_url": user["linkedin_url"]})
            analysis = model.invoke(
                LINKEDIN_ANALYSIS_PROMPT.format(
                    model_schema=json.dumps(LinkedInAnalysis.model_json_schema(), indent=2),
                    linkedin_data=json.dumps(linkedin_data["data"], indent=2),
                    target_role=state['config']['target_role']
                )
            )
            
            enriched_user = {
                **user,
                "linkedin_raw": linkedin_data["data"],
                "linkedin_analysis": analysis.dict(),
                "sources": user["sources"] + ["linkedin_analyzed"]
            }
            enriched.append(enriched_user)
            
            messages.append(
                ToolMessage(
                    tool_call_id=f"linkedin_{user['email']}",
                    tool_name="linkedin",
                    content=f"Beriket {user['email']}"
                )
            )
        except Exception as e:
            messages.append(
                ToolMessage(
                    tool_call_id=f"linkedin_error_{user['email']}",
                    tool_name="linkedin",
                    content=f"Feil: {str(e)}"
                )
            )
    
    return {
        "messages": messages,
        "users": enriched or state["users"]
    }

def handle_error(state: AgentState, error: Exception) -> Dict:
    """Enkel feilhåndtering"""
    return {
        "messages": [
            ToolMessage(
                tool_call_id="error",
                tool_name="error_handler",
                content=f"Feil oppstod: {str(error)}"
            )
        ]
    }

# Oppdater workflow
def create_workflow() -> StateGraph:
    """Oppretter workflow."""
    workflow = StateGraph(AgentState)
    
    # Legg til noder og tools
    workflow.add_node("collect", collect_hunter_data)
    workflow.add_node("prioritize", prioritize_users)
    workflow.add_node("enrich", get_linkedin_info)
    workflow.add_node("tools", ToolNode([linkedin_tool, hunter_tool]))
    
    # Definer flyt
    workflow.add_edge("__start__", "collect")
    workflow.add_edge("tools", "enrich")
    
    # Betinget routing
    workflow.add_conditional_edges(
        "collect",
        lambda s: "prioritize" if s["users"] else "__end__"
    )
    workflow.add_conditional_edges(
        "prioritize",
        lambda s: "enrich" if any(u.get("priority_score", 0) > 0 
                                 for u in s["users"]) else "__end__"
    )
    workflow.add_conditional_edges("enrich", tools_condition)
    
    return workflow.compile()

def get_config() -> RunnableConfig:
    return RunnableConfig(
        callbacks=[],
        tags=["prospect-agent"],
        metadata={"version": "1.0"}
    )

# Compile workflow
app = create_workflow()

# Test
if __name__ == "__main__":
    result = app.invoke({
        "messages": [],
        "users": [],
        "config": SearchConfig(
            domain="documaster.com",
            target_role="marketing",
            max_results=3
        )
    })

__all__ = ['app', 'get_config']  # Fjern User fra exports