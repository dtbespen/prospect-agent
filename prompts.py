import json
from models import LinkedInAnalysis, PriorityAnalysis, LinkedInRawData

def get_example_linkedin_analysis():
    """Generer et eksempel basert på LinkedInAnalysis modellen"""
    return {
        field: field_info.description or "Eksempel verdi"
        for field, field_info in LinkedInAnalysis.model_fields.items()
    }

def get_example_priority_analysis():
    """Generer et eksempel basert på PriorityAnalysis modellen"""
    return {
        "users": {
            "person@example.com": {
                "score": 0.85,
                "reason": "Detaljert begrunnelse på norsk"
            }
        }
    }

LINKEDIN_ANALYSIS_PROMPT = """
Analyser denne LinkedIn profilen og map dataen til følgende modell:

{model_schema}

Rå LinkedIn data:
{linkedin_data}

Målrolle som skal vurderes:
{target_role}

VIKTIG:
- Returner et JSON-objekt som følger modellen over
- Bruk alle tilgjengelige felter fra rådataen
- Behold original datatype (int, string, etc.)
Instruksjoner for mapping:
- Beregn years_in_role basert på current_job_duration eller start_year/month
- Beregn total_experience_years ved å summere varighet av alle experiences
- Utled seniority_level basert på erfaring og roller (junior/senior/lead/etc.)
- Sett role_relevance til en score mellom 0-1 basert på match med målrollen
- Lag en kort, presis overall_summary på norsk
- Inkluder bare de mest relevante key_skills (maks 5)
- List opp unike industries fra all erfaring
"""

PRIORITY_ANALYSIS_PROMPT = """
Analyser disse personene for å finne de som har rollen: {role}.

Personer å vurdere:
{users}

VIKTIG: Returner en liste som følger PriorityAnalysis-modellen:
{model_schema}

Vurder hver person basert på:
- Hvor godt nåværende rolle matcher målrollen
- Tydelighet i rollebeskrivelsen
- Relevans av arbeidssted/bransje

Returner maksimalt {max_results} personer, sortert etter score.
""" 