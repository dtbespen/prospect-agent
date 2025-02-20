from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from agent import app as workflow_app, get_config
from models import SearchConfig, EnrichedUser
from agent import AgentState

class ProspectRequest(BaseModel):
    domain: str
    target_role: str
    max_results: Optional[int] = 5
    search_depth: Optional[int] = 1

class ProspectResponse(BaseModel):
    users: List[EnrichedUser]
    message: str

app = FastAPI(
    title="Prospect Agent API",
    description="API for å finne og analysere relevante kontakter",
    version="1.0.0"
)

@app.post("/prospects", response_model=ProspectResponse)
async def find_prospects(request: ProspectRequest):
    """Finn og analyser relevante kontakter basert på domene og målrolle."""
    try:
        result = workflow_app.invoke(AgentState(
            messages=[],
            config=SearchConfig(
                domain=request.domain,
                target_role=request.target_role,
                max_results=request.max_results,
                search_depth=request.search_depth
            ),
            users=[]
        ), config=get_config())
        
        # Filtrer ut bare de som er ferdig analysert
        analyzed_users = [
            EnrichedUser(**user).ser_model()  # Bruk ser_model for flat struktur
            for user in result["users"] 
            if "linkedin_analyzed" in user.get("sources", [])
        ]
        
        return ProspectResponse(
            users=analyzed_users,
            message=f"Fant {len(analyzed_users)} relevante kontakter med full analyse"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feil under prosessering: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Velkommen til Prospect Agent API"} 