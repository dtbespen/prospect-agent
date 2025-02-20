from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from models import UserBase

def add_messages(old_messages: List[BaseMessage], new_messages: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer for å kombinere meldinger"""
    return old_messages + new_messages

def add_users(old_users: List[Dict], new_users: List[Dict]) -> List[Dict]:
    """Reducer for å kombinere og deduplisere brukere"""
    seen_emails = {u.get("email") for u in old_users}
    unique_new = [u for u in new_users if u.get("email") not in seen_emails]
    return old_users + unique_new 