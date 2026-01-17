import sys
import os
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import ProjectConfig

logger = logging.getLogger("ReceiptBridge")

# This file previously contained SQLAlchemy models, database operations,
# and business logic related to receipts and pantry management.
# This functionality has been refactored into:
# - database/repositories/product_repo.py (SQLAlchemy models and data access)
# - core/services/pantry_service.py (Business logic)
# - adapters/obsidian/view_generator.py (Markdown generation for pantry state)
# - adapters/obsidian/action_processor.py (Processing of command notes)
# - Other functionalities related to receipt processing in utils/receipt_agents

# No code should remain here that directly interacts with the database
# or performs complex business logic related to receipts or pantry.
# This file is effectively deprecated and will be removed or repurposed
# in future phases as the system transitions to the new architecture.

# Remaining imports and logger are kept for compatibility during transition
# and can be removed once all dependencies on this file are resolved.