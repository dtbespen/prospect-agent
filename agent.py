from typing import Annotated, List, Dict, Union
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

# 3. SCREENING NODE (FØRSTE STEG)
class HunterDataCollector:
    """Samler kontakter fra Hunter.io API"""
    
    @traceable(run_type="chain", name="hunter_collection")
    def run(self, state: AgentState, config: RunnableConfig) -> Dict[str, Union[MessageList, UserList]]:
        """Henter alle kontakter fra Hunter.io"""
        messages = []
        users = []
        
        try:
            # Valider hunter data først
            hunter_data = HunterResponse(**hunter_tool.invoke({
                "domain": state["config"]["domain"],
                "api_key": os.getenv("HUNTER_API_KEY")
            }))
            
            # Konverter til UserBase
            users = [
                UserBase(
                    email=email["value"],
                    **{k: email.get(v) for k, v in {
                        'first_name': 'first_name',
                        'last_name': 'last_name',
                        'role': 'position',
                        'linkedin_url': 'linkedin',
                        'phone_number': 'phone_number'
                    }.items()},
                    confidence=str(email.get('confidence', '')),  # Konverter til string
                    sources=["hunter"]
                ).model_dump()
                for email in hunter_data.emails
            ]
            
        except Exception as e:
            messages.append(
                ToolMessage(
                    tool_call_id="hunter_error",
                    tool_name="hunter_collection",
                    content=f"Feil under henting av kontakter: {str(e)}"
                )
            )
        
        # Returner bare det som skal oppdateres
        return {
            "messages": messages,  # Annotated[List[BaseMessage], "messages"]
            "users": users        # Annotated[List[Dict], "users"]
        }

# 4. PRIORITERING NODE (ANDRE STEG)
@traceable(
    run_type="chain",
    name="prioritize_users",
    metadata={"type": "prioritization"}
)
def prioritize_users(state: AgentState, config: RunnableConfig) -> Dict[str, Union[MessageList, UserList]]:
    """Prioriterer brukere basert på deres egnethet for målrollen."""
    
    # Setup LLM med strukturert output
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY")  # Bruk API-nøkkel direkte
    ).with_structured_output(
        PriorityAnalysis,
        method="json_mode"
    )
    
    # Filtrer brukere med rolle
    users_to_analyze = [
        UserBase(**u) for u in state["users"] 
        if u.get("role") is not None  # Sjekk for None siden role er Optional
    ]
    
    if not users_to_analyze:
        return {
            "messages": [HumanMessage(content="Ingen brukere med roller funnet")],
            "users": []
        }
    
    # Analyser alle brukere i én forespørsel
    analysis = model.invoke(
        PRIORITY_ANALYSIS_PROMPT.format(
            example=json.dumps(get_example_priority_analysis(), indent=2),
            role=state['config']['target_role'],
            users=json.dumps([u.model_dump() for u in users_to_analyze], indent=2),
            max_results=state['config'].get('max_results', 5)
        )
    )
    
    # Oppdater brukere med prioriteringer
    prioritized = []
    for user in users_to_analyze:
        analysis_result = analysis.users.get(user.email)
        if analysis_result:  # Ta med alle som ble valgt av LLM
            user.priority_score = analysis_result["score"]
            user.priority_reason = analysis_result["reason"]
            user.sources = user.sources + ["prioritized"]
            prioritized.append(user.model_dump())
    
    return {
        "messages": [HumanMessage(content=f"Analyserte {len(users_to_analyze)} brukere, prioriterte {len(prioritized)}")],
        "users": prioritized  # Bare de prioriterte brukerne går videre
    }

@traceable(run_type="chain", name="get_linkedin_info")
def get_linkedin_info(state: AgentState, config: RunnableConfig) -> Dict[str, Union[MessageList, UserList]]:
    """Beriker prioriterte brukere med LinkedIn data og analyse."""
    
    # Finn prioriterte brukere med LinkedIn URL
    prioritized_users = [
        UserBase(**u) for u in state["users"]
        if "prioritized" in u.get("sources", [])
        and u.get("linkedin_url")
        and (u.get("priority_score") or 0) > 0
    ]
    
    if not prioritized_users:
        return {
            "messages": [HumanMessage(content="Ingen brukere nådde LinkedIn-terskelen")],
            "users": state["users"]  # Behold eksisterende brukere uendret
        }
    
    # Setup LLM med strukturert output
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(LinkedInAnalysis)
    
    enriched = []
    analysis_messages = []
    
    for user in prioritized_users:
        try:
            # 1. Hent LinkedIn data
            linkedin_data = linkedin_tool.invoke({
                "linkedin_url": user.linkedin_url
            })
            raw_data = LinkedInRawData(data=linkedin_data["data"])
            
            # 2. La LLM analysere dataene
            try:
                analysis = model.invoke(
                    LINKEDIN_ANALYSIS_PROMPT.format(
                        model_schema=json.dumps(LinkedInAnalysis.model_json_schema(), indent=2),
                        linkedin_data=json.dumps(linkedin_data["data"], indent=2),
                        target_role=state['config']['target_role']
                    )
                )
            except Exception as e:
                analysis_messages.append(
                    ToolMessage(
                        tool_call_id=f"linkedin_analysis_error_{user.email}",
                        tool_name="analyze_linkedin",
                        content=f"Feil under analyse av LinkedIn-data: {str(e)}\nRå feil: {e.__class__.__name__}"
                    )
                )
                continue
            
            # 3. Opprett beriket bruker og serialiser flat
            try:
                enriched_user = EnrichedUser(
                    **{k: v for k, v in user.model_dump().items() if k != 'sources'},
                    linkedin_raw=raw_data,
                    linkedin_analysis=analysis,
                    sources=user.sources + ["linkedin_analyzed"]
                )
                # Bruk model_dump() istedenfor ser_model()
                enriched.append(enriched_user.model_dump())
            except Exception as e:
                logging.error(f"Serialiseringsfeil for {user.email}: {str(e)}")
                continue
            
            analysis_messages.append(
                ToolMessage(
                    tool_call_id=f"linkedin_analysis_{user.email}",
                    tool_name="analyze_linkedin",
                    content=f"Analyserte LinkedIn-profil for {user.email}"
                )
            )
            
        except Exception as e:
            logging.error(f"LinkedIn-datafeil for {user.email}: {str(e)}")
            continue
    
    return {
        "messages": analysis_messages,
        "users": enriched
    }

# Workflow setup og kompilering -> # Arbeidsflyt oppsett og kompilering
def create_workflow() -> StateGraph:
    """Oppretter og konfigurerer workflow."""
    
    graph_builder = StateGraph(AgentState)
    
    # Initialiser collector
    hunter_collector = HunterDataCollector()
    
    # Legg til noder
    graph_builder.add_node("hunter_collection", hunter_collector.run)
    graph_builder.add_node("prioritize_users", prioritize_users)
    graph_builder.add_node("get_linkedin_info", get_linkedin_info)
    
    # Definer flyten
    graph_builder.add_edge(START, "hunter_collection")
    graph_builder.add_edge("hunter_collection", "prioritize_users")
    graph_builder.add_edge("prioritize_users", "get_linkedin_info")
    
    return graph_builder

def get_config() -> RunnableConfig:
    """Konfigurasjon for kjøring og sporing."""
    # Fjern ConsoleCallbackHandler for å redusere logging
    return RunnableConfig(
        callbacks=[],  # Fjernet ConsoleCallbackHandler
        tags=["test", "prospect-agent"],
        metadata={
            "version": "1.0",
            "model_name": DEFAULT_MODEL
        }
    )

# Compile workflow
app = create_workflow().compile()

# Test
if __name__ == "__main__":
    config = SearchConfig(
        domain="documaster.com",
        target_role="ansvarlig for digital eller markedsføring",
        search_depth=1,
        max_results=3
    )
    
    try:
        result = app.invoke(AgentState(
            messages=[],
            config=config,
            users=[]
        ))
        
        # Kommenter ut unødvendige loggmeldinger
        # logging.info("Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Brukere med LinkedIn-analyse")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: {user.first_name} {user.last_name} ({user.email})")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Nåværende rolle: {user.get('analysis_job_title')} hos {user.get('analysis_company')}")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Erfaring: {user.get('analysis_total_experience_years', 0)} år totalt, {user.get('analysis_years_in_role', 0)} år i nåværende rolle")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Nivå: {user.get('analysis_seniority_level', 'Ukjent')}")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Relevans for {config['target_role']}: {user.get('analysis_role_relevance', 0):.2f}")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Ferdigheter: {', '.join(user.get('analysis_key_skills', []))}")
        # logging.info(f"Node: get_linkedin_info, Agent: LinkedIn, Tool: Analysis, Message: Oppsummering: {user.get('analysis_overall_summary', 'Ingen oppsummering')}")
        
    finally:
        # Ensure all traces are submitted
        wait_for_all_tracers()

__all__ = ['app', 'get_config']  # Fjern User fra exports