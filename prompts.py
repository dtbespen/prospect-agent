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

VIKTIG: Returner et JSON-objekt som følger PriorityAnalysis modellen.
Eksempel struktur:
{example}

For hver person, gjør en helhetlig vurdering basert på:

KRITISKE FAKTORER:
- Rolle-match: Har personen en stilling/tittel som matcher målrollen vi leter etter?
- LinkedIn-profil: Må ha en gyldig LinkedIn URL for videre verifisering
- Data-kvalitet: Vurder confidence-score og mengde tilgjengelig informasjon

VURDERING:
- Gi en score fra 0.0 til 1.0 basert på hvor godt nåværende rolle matcher
- Høyere score til personer som har en rolle/tittel som direkte matcher det vi leter etter
- Lavere score til personer med roller som er uklare eller ikke relevante
- Personer uten LinkedIn-profil skal automatisk få score 0.0

Returner de {max_results} mest relevante personene.

NB: 
- Vi leter etter personer som HAR denne rollen nå, ikke potensielle kandidater
- Prioriter direkte rolle-match over andre faktorer
- Begrunn tydelig hvorfor hver persons nåværende rolle er relevant
- Skriv all begrunnelse på norsk
""" 