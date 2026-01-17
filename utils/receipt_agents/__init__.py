"""
Receipt Agents - Shop-specific preprocessing for OCR text.

Each agent handles the unique formatting quirks of different stores.
"""

from .base import BaseReceiptAgent
from .biedronka import BiedronkaAgent
from .lidl import LidlAgent
from .auchan import AuchanAgent
from .zabka import ZabkaAgent

# Shop name to agent class mapping
SHOP_AGENTS = {
    "Biedronka": BiedronkaAgent,
    "Lidl": LidlAgent,
    "Auchan": AuchanAgent,
    "Zabka": ZabkaAgent,
}


def detect_shop(text: str) -> str:
    """
    Detect shop name from OCR text.

    Centralized shop detection to avoid duplication across modules.

    Args:
        text: Raw OCR text from receipt

    Returns:
        Shop name string (e.g., "Biedronka", "Lidl", "Zabka", "Sklep")
    """
    text_lower = text.lower()

    if "biedronka" in text_lower or "jeronimo" in text_lower:
        return "Biedronka"
    if "lidl" in text_lower:
        return "Lidl"
    if "auchan" in text_lower:
        return "Auchan"
    if "żabka" in text_lower or "zabka" in text_lower:
        return "Zabka"
    if "kaufland" in text_lower:
        return "Kaufland"
    if "rossmann" in text_lower:
        return "Rossmann"
    if "carrefour" in text_lower:
        return "Carrefour"
    if "dino" in text_lower:
        return "Dino"

    return "Sklep"


def get_agent(shop_name: str) -> BaseReceiptAgent:
    """
    Get shop-specific agent instance.

    Args:
        shop_name: Shop name from detect_shop()

    Returns:
        Appropriate agent instance, defaults to BiedronkaAgent
    """
    agent_class = SHOP_AGENTS.get(shop_name, BiedronkaAgent)
    return agent_class()


__all__ = [
    'BaseReceiptAgent',
    'BiedronkaAgent',
    'LidlAgent',
    'AuchanAgent',
    'ZabkaAgent',
    'SHOP_AGENTS',
    'detect_shop',
    'get_agent',
]
