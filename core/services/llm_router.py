"""
Smart Router - Intelligent LLM Provider Selection for Hybrid AI System

Routes tasks to optimal providers based on:
- Task type (receipt parsing, RAG, summarization, etc.)
- Complexity estimation
- Provider availability
- Cost constraints
- Performance requirements

Architecture:
    SmartRouter → TaskClassifier → ProviderSelector → Execute with Fallback
"""

import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import ollama
import openai
# Use google-genai (modern SDK)
# Migration from google.generativeai complete
from google import genai
from google.genai import types

import yaml
from pathlib import Path

from config import ProjectConfig
from utils.ollama_optimizer import (
    ollama_chat_optimized,
    TaskType as OptimizerTaskType,
    get_optimal_options,
    get_model_config
)

logger = logging.getLogger("SmartRouter")


class TaskType(Enum):
    """Classification of AI tasks for routing decisions."""
    RECEIPT_PARSING = "receipt_parsing"      # JSON extraction from OCR
    RAG_CHAT = "rag_chat"                    # Q&A over knowledge base
    RAG_HYDE = "rag_hyde"                    # Hypothetical document generation
    SUMMARIZATION = "summarization"          # Article/text summarization
    TAGGING = "tagging"                      # Note tagging and categorization
    OCR_VISION = "ocr_vision"                # Image analysis (multimodal)
    NOTE_REFINEMENT = "note_refinement"      # Link suggestions, metadata
    EMAIL_TRIAGE = "email_triage"            # Email classification and action decision
    GENERAL = "general"                      # Fallback for unclassified tasks


class Provider(Enum):
    """Available LLM providers."""
    LOCAL_FAST = "local_fast"        # Ollama - gemma3:4b (quick tasks)
    LOCAL_SMART = "local_smart"      # Ollama - Bielik 11B (complex Polish)
    LOCAL_REASONING = "local_reason" # Ollama - DeepSeek-R1 (JSON/reasoning)
    OPENAI = "openai"                # OpenAI - gpt-4o-mini
    GEMINI = "gemini"                # Google - gemini-2.0-flash-lite


class Complexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Short prompts, simple outputs
    MEDIUM = "medium"      # Moderate context, structured output
    COMPLEX = "complex"    # Long context, multi-step reasoning


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    primary: Provider
    fallback: Provider
    model_name: str
    reason: str


@dataclass
class ExecutionResult:
    """Result of task execution."""
    success: bool
    content: Optional[str]
    provider_used: Provider
    model_used: str
    latency_ms: float
    tokens_estimated: int = 0
    error: Optional[str] = None


class ProviderHealth:
    """Tracks provider availability and performance."""

    def __init__(self):
        self._status: Dict[Provider, bool] = {p: True for p in Provider}
        self._last_check: Dict[Provider, datetime] = {}
        self._failure_count: Dict[Provider, int] = {p: 0 for p in Provider}
        self._check_interval_seconds = 60

    def is_available(self, provider: Provider) -> bool:
        """Check if provider is currently available."""
        last = self._last_check.get(provider)
        if last and (datetime.now() - last).seconds < self._check_interval_seconds:
            return self._status[provider]

        # Perform health check
        available = self._check_health(provider)
        self._status[provider] = available
        self._last_check[provider] = datetime.now()

        if available:
            self._failure_count[provider] = 0

        return available

    def _check_health(self, provider: Provider) -> bool:
        """Perform actual health check for provider."""
        try:
            if provider in [Provider.LOCAL_FAST, Provider.LOCAL_SMART, Provider.LOCAL_REASONING]:
                ollama.list()
                return True
            elif provider == Provider.OPENAI:
                return bool(ProjectConfig.OPENAI_API_KEY)
            elif provider == Provider.GEMINI:
                return bool(ProjectConfig.GEMINI_API_KEY)
        except Exception as e:
            logger.warning(f"Health check failed for {provider.value}: {e}")
            return False
        return False

    def report_failure(self, provider: Provider):
        """Report a failure for circuit breaker logic."""
        self._failure_count[provider] = self._failure_count.get(provider, 0) + 1
        if self._failure_count[provider] >= 3:
            self._status[provider] = False
            logger.warning(f"Provider {provider.value} marked unavailable after 3 failures")

    def report_success(self, provider: Provider):
        """Report success to reset failure counter."""
        self._failure_count[provider] = 0
        self._status[provider] = True


class SmartRouter:
    """
    Intelligent router for hybrid LLM system.

    Routes tasks to optimal providers based on task type, complexity,
    availability, and cost constraints.

    Usage:
        router = SmartRouter()
        result = router.execute(TaskType.RECEIPT_PARSING, prompt, format="json")
    """

    # Model mapping per provider
    MODEL_MAP: Dict[Provider, str] = {
        Provider.LOCAL_FAST: ProjectConfig.OLLAMA_MODEL_FAST,
        Provider.LOCAL_SMART: ProjectConfig.OLLAMA_MODEL,
        Provider.LOCAL_REASONING: "deepseek-r1:latest",
        Provider.OPENAI: ProjectConfig.OPENAI_MODEL,
        Provider.GEMINI: ProjectConfig.GEMINI_MODEL,
    }

    # TaskType -> OptimizerTaskType mapping for Ollama optimization
    OPTIMIZER_TASK_MAP: Dict[TaskType, OptimizerTaskType] = {
        TaskType.RECEIPT_PARSING: OptimizerTaskType.JSON_EXTRACT,
        TaskType.RAG_CHAT: OptimizerTaskType.CHAT,
        TaskType.RAG_HYDE: OptimizerTaskType.HYDE,
        TaskType.SUMMARIZATION: OptimizerTaskType.SUMMARIZATION,
        TaskType.TAGGING: OptimizerTaskType.TAGGING,
        TaskType.OCR_VISION: OptimizerTaskType.CHAT,  # Vision handled separately
        TaskType.NOTE_REFINEMENT: OptimizerTaskType.JSON_EXTRACT,
        TaskType.EMAIL_TRIAGE: OptimizerTaskType.JSON_EXTRACT,  # Returns JSON action
        TaskType.GENERAL: OptimizerTaskType.CHAT,
    }

    def __init__(self):
        self.health = ProviderHealth()
        self._stats: Dict[str, int] = {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "fallback_used": 0,
        }
        self.routing_table: Dict[Tuple[TaskType, Complexity], Tuple[Provider, Provider, str]] = {}
        self.load_routing_config()

        self._init_clients()
        logger.info("SmartRouter initialized")

    def _init_clients(self):
        """Initialize API clients."""
        # OpenAI client
        if ProjectConfig.OPENAI_API_KEY:
            self._openai_client = openai.OpenAI(api_key=ProjectConfig.OPENAI_API_KEY)
        else:
            self._openai_client = None

        # Gemini client
        if ProjectConfig.GEMINI_API_KEY:
            try:
                self._gemini_client = genai.Client(api_key=ProjectConfig.GEMINI_API_KEY)
                self._gemini_model_name = ProjectConfig.GEMINI_MODEL
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self._gemini_client = None
        else:
            self._gemini_client = None

    def estimate_complexity(self, prompt: str, context_length: int = 0) -> Complexity:
        """
        Estimate task complexity based on prompt characteristics.

        Args:
            prompt: The user prompt
            context_length: Additional context length (e.g., RAG documents)

        Returns:
            Complexity level
        """
        total_length = len(prompt) + context_length

        if total_length < 500:
            return Complexity.SIMPLE
        elif total_length < 3000:
            return Complexity.MEDIUM
        else:
            return Complexity.COMPLEX

    def load_routing_config(self):
        """Load routing configuration from YAML."""
        config_path = ProjectConfig.BASE_DIR / "config" / "routing.yaml"
        if not config_path.exists():
            logger.warning("Routing config not found, using empty table")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            new_table = {}
            for route in data.get("routes", []):
                try:
                    task = TaskType(route["task"])
                    complexity = Complexity(route["complexity"])
                    primary = Provider(route["primary"])
                    fallback = Provider(route["fallback"])
                    reason = route.get("reason", "Configured route")
                    
                    new_table[(task, complexity)] = (primary, fallback, reason)
                except ValueError as e:
                    logger.warning(f"Invalid route entry: {e}")
                    
            self.routing_table = new_table
            logger.info(f"Loaded {len(self.routing_table)} routes from config")
            
        except Exception as e:
            logger.error(f"Failed to load routing config: {e}")

    def route(
        self,
        task_type: TaskType,
        prompt: str,
        complexity: Optional[Complexity] = None,
        context_length: int = 0
    ) -> RoutingDecision:
        """
        Determine optimal provider for task.

        Args:
            task_type: Type of AI task
            prompt: User prompt
            complexity: Optional explicit complexity (auto-detected if None)
            context_length: Additional context length

        Returns:
            RoutingDecision with primary and fallback providers
        """
        if complexity is None:
            complexity = self.estimate_complexity(prompt, context_length)

        # Get routing from table
        key = (task_type, complexity)
        if key not in self.routing_table:
            # Fallback to general if specific route not found
            key = (TaskType.GENERAL, complexity)
            
        if key in self.routing_table:
            primary, fallback, reason = self.routing_table[key]
        else:
            # Absolute fallback if completely missing
            primary = Provider.LOCAL_SMART
            fallback = Provider.OPENAI
            reason = "Default fallback (no route found)"

        # Check availability and swap if needed
        if not self.health.is_available(primary):
            logger.info(f"Primary {primary.value} unavailable, using fallback {fallback.value}")
            primary, fallback = fallback, primary
            reason = f"Fallback: {reason}"

        return RoutingDecision(
            primary=primary,
            fallback=fallback,
            model_name=self.MODEL_MAP[primary],
            reason=reason
        )

    def execute(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: Optional[str] = None,
        format: Optional[str] = None,
        complexity: Optional[Complexity] = None,
        context_length: int = 0,
        timeout: float = 60.0
    ) -> ExecutionResult:
        """
        Execute task with automatic provider selection and fallback.

        Args:
            task_type: Type of AI task
            prompt: User prompt
            system_prompt: Optional system instructions
            format: Output format ("json" for structured output)
            complexity: Optional explicit complexity
            context_length: Additional context length for routing
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with content or error
        """
        self._stats["total_requests"] += 1

        decision = self.route(task_type, prompt, complexity, context_length)
        logger.info(f"Routing {task_type.value} to {decision.primary.value} ({decision.reason})")

        # Try primary provider
        result = self._execute_on_provider(
            decision.primary,
            prompt,
            system_prompt,
            format,
            timeout,
            task_type
        )

        if result.success:
            self.health.report_success(decision.primary)
            self._track_provider(decision.primary)
            return result

        # Fallback
        logger.warning(f"Primary failed: {result.error}, trying fallback {decision.fallback.value}")
        self.health.report_failure(decision.primary)
        self._stats["fallback_used"] += 1

        result = self._execute_on_provider(
            decision.fallback,
            prompt,
            system_prompt,
            format,
            timeout,
            task_type
        )

        if result.success:
            self.health.report_success(decision.fallback)
            self._track_provider(decision.fallback)
        else:
            self.health.report_failure(decision.fallback)

        return result

    def _execute_on_provider(
        self,
        provider: Provider,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str],
        timeout: float,
        task_type: TaskType = TaskType.GENERAL
    ) -> ExecutionResult:
        """Execute request on specific provider."""
        start_time = time.time()
        model_name = self.MODEL_MAP[provider]

        try:
            if provider in [Provider.LOCAL_FAST, Provider.LOCAL_SMART, Provider.LOCAL_REASONING]:
                content = self._call_ollama(model_name, prompt, system_prompt, format, task_type)
            elif provider == Provider.OPENAI:
                content = self._call_openai(prompt, system_prompt, format)
            elif provider == Provider.GEMINI:
                content = self._call_gemini(prompt, system_prompt, format)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            latency = (time.time() - start_time) * 1000
            tokens = len(prompt.split()) + len((content or "").split())

            return ExecutionResult(
                success=True,
                content=content,
                provider_used=provider,
                model_used=model_name,
                latency_ms=latency,
                tokens_estimated=tokens
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Provider {provider.value} error: {e}")
            return ExecutionResult(
                success=False,
                content=None,
                provider_used=provider,
                model_used=model_name,
                latency_ms=latency,
                error=str(e)
            )

    def _call_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str],
        task_type: TaskType = TaskType.GENERAL
    ) -> Optional[str]:
        """Call Ollama local model with optimized parameters."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Map to optimizer task type for parameter tuning
        optimizer_task = self.OPTIMIZER_TASK_MAP.get(task_type, OptimizerTaskType.CHAT)

        # Use optimized call with model-specific and task-specific parameters
        response = ollama_chat_optimized(
            model=model,
            messages=messages,
            task_type=optimizer_task,
            format="json" if format == "json" else None,
            timeout=120.0
        )
        
        # Enforce structured output for Bielik if format is json
        # Since we want specific schema {intent: ..., params: ...}
        # We can validate it here or assume the prompt handles it.
        # But for 'Bielik' specifically, we might want to ensure JSON.
        
        content = response["message"]["content"]
        
        if format == "json":
            try:
                # Try simple clean up if markdown blocks used
                clean_content = content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content.split("```json")[1]
                if clean_content.endswith("```"):
                     clean_content = clean_content.rsplit("```", 1)[0]
                
                # Check if valid JSON
                json.loads(clean_content)
                return clean_content
            except Exception:
                logger.warning(f"Ollama response was not valid JSON: {content[:50]}...")
                # Fallback or return as is (caller handles error)
        
        return content

    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str]
    ) -> Optional[str]:
        """Call OpenAI API."""
        if not self._openai_client:
            raise ValueError("OpenAI client not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=ProjectConfig.OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"} if format == "json" else None
        )
        return response.choices[0].message.content

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str]
    ) -> Optional[str]:
        """Call Gemini API."""
        if not self._gemini_client:
            raise ValueError("Gemini client not initialized")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json" if format == "json" else "text/plain"
        )

        response = self._gemini_client.models.generate_content(
            model=self._gemini_model_name,
            contents=prompt,
            config=config
        )
        return response.text

    def _track_provider(self, provider: Provider):
        """Track provider usage for stats."""
        if provider in [Provider.LOCAL_FAST, Provider.LOCAL_SMART, Provider.LOCAL_REASONING]:
            self._stats["local_requests"] += 1
        else:
            self._stats["cloud_requests"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "local_percentage": (self._stats["local_requests"] / total * 100) if total > 0 else 0,
            "cloud_percentage": (self._stats["cloud_requests"] / total * 100) if total > 0 else 0,
            "fallback_rate": (self._stats["fallback_used"] / total * 100) if total > 0 else 0,
        }

    def get_provider_status(self) -> Dict[str, bool]:
        """Get availability status of all providers."""
        return {p.value: self.health.is_available(p) for p in Provider}


# Singleton instance for easy import
_router_instance: Optional[SmartRouter] = None

def get_router() -> SmartRouter:
    """Get or create singleton SmartRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter()
    return _router_instance
