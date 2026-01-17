import google.genai as genai
import openai
import ollama
import logging
import json
import asyncio
from typing import Optional, Union, List, Any, Dict
from pathlib import Path
import PIL.Image
from config import ProjectConfig, logger

# Smart Router integration (lazy import to avoid circular dependency)
_smart_router = None

def _get_smart_router():
    """Lazy load SmartRouter to avoid circular imports."""
    global _smart_router
    if _smart_router is None and ProjectConfig.SMART_ROUTER_ENABLED:
        try:
            from core.services.llm_router import get_router
            _smart_router = get_router()
        except ImportError:
            logger.warning("SmartRouter not available, using legacy routing")
    return _smart_router

class UniversalBrain:
    """
    Unified Bridge for Multiple AI Providers: Local (Ollama), Gemini, and OpenAI.
    Allows seamless switching between models based on cost/performance needs.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or ProjectConfig.AI_PROVIDER
        self.available = False
        
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "local":
            self._init_local()
        else:
            logger.error(f"Unknown AI Provider: {self.provider}")

    def _init_gemini(self):
        if not ProjectConfig.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY missing. Gemini disabled.")
            return
        try:
            genai.configure(api_key=ProjectConfig.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(ProjectConfig.GEMINI_MODEL)
            self.available = True
            logger.info(f"UniversalBrain (Gemini) initialized: {ProjectConfig.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to init Gemini: {e}")

    def _init_openai(self):
        if not ProjectConfig.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY missing. OpenAI disabled.")
            return
        try:
            self.client = openai.OpenAI(api_key=ProjectConfig.OPENAI_API_KEY)
            self.model_name = ProjectConfig.OPENAI_MODEL
            self.available = True
            logger.info(f"UniversalBrain (OpenAI) initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to init OpenAI: {e}")

    def _init_local(self):
        try:
            # Simple check if ollama is reachable
            ollama.list()
            self.model_name = ProjectConfig.OLLAMA_MODEL
            self.available = True
            logger.info(f"UniversalBrain (Local/Ollama) initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to init Local Ollama: {e}")

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None, format: Optional[str] = None, model_name: Optional[str] = None) -> Optional[str]:

        """
        Generates text content from the selected provider.
        Supports optional system_prompt for better instruction following.
        """
        if not self.available:
            return None

        try:
            if self.provider == "gemini":
                # For Gemini, we can recreate the model with system_instruction or just prepend
                # Recreating is safer for static instructions
                model = self.model
                if system_prompt:
                    model = genai.GenerativeModel(
                        model_name=ProjectConfig.GEMINI_MODEL,
                        system_instruction=system_prompt
                    )
                
                generation_config = {}
                if format == "json":
                    generation_config["response_mime_type"] = "application/json"
                
                response = model.generate_content(prompt, generation_config=generation_config)
                return response.text
            
            elif self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"} if format == "json" else None
                )
                return response.choices[0].message.content
            
            elif self.provider == "local":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                target_model = model_name or self.model_name
                response = ollama.chat(
                    model=target_model,
                    messages=messages,
                    format='json' if format == 'json' else None
                )
                content = response['message']['content']
                
                # DeepSeek-R1 specific: remove <think> tags
                if "deepseek" in target_model.lower() and "<think>" in content:
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    # DeepSeek sometimes outputs markdown code blocks around JSON
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```', '', content).strip()
                    
                return content

        except Exception as e:
            logger.error(f"UniversalBrain ({self.provider}) Error: {e}")
            return None

    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Optional[str]:
        """
        Multimodal analysis (Image + Text).
        Note: Local Ollama needs specific vision models (e.g. llava).
        """
        if not self.available:
            return None

        try:
            if self.provider == "gemini":
                img = PIL.Image.open(image_path)
                response = self.model.generate_content([prompt, img])
                return response.text
            
            elif self.provider == "openai":
                # OpenAI requires base64 for images
                import base64
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                },
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content
                
            elif self.provider == "local":
                # Assuming user has a vision-capable model for local
                with open(image_path, 'rb') as f:
                    response = ollama.chat(
                        model="llama3.2-vision", # Hardcoded or configurable for local vision
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [f.read()]
                        }]
                    )
                return response['message']['content']

        except Exception as e:
            logger.error(f"UniversalBrain Image Error: {e}")
            return None

    def refine_note(self, content: str, existing_tags: List[str] = None, existing_titles: List[str] = None) -> Dict[str, Any]:
        """
        Cloud-based note refinement: tagging, linking, and metadata.
        """
        if not self.available:
            return {"tags": [], "links": [], "summary": ""}

        tags_str = ", ".join(existing_tags[:100]) if existing_tags else "brak"
        titles_str = ", ".join(existing_titles[:200]) if existing_titles else "brak"

        system_prompt = ProjectConfig.PROMPTS.get('refine_note', {}).get('system_prompt', "Jesteś ekspertem od zarządzania wiedzą (PKM) w Obsidian.")
        
        user_prompt = f"""
        Twoim zadaniem jest wzbogacenie poniższej notatki.

        ZASADY:
        1. TAGI: Wybierz 3-5 tagów. Zainspiruj się istniejącymi: [{tags_str}].
        2. LINKI: Zaproponuj 3-5 linków wewnętrznych [[Nazwa Notatki]] do istniejących tematów: [{titles_str}].
        3. PODSUMOWANIE: Napisz jedno zdanie podsumowania.

        TREŚĆ NOTATKI:
        {content[:15000]}

        Odpowiedz TYLKO JSON:
        {{
          "tags": ["tag1", "tag2"],
          "links": ["Tytuł 1", "Tytuł 2"],
          "summary": "To jest podsumowanie..."
        }}
        """

        try:
            res_text = self.generate_content(user_prompt, system_prompt=system_prompt, format="json")
            return json.loads(res_text)
        except Exception as e:
            logger.error(f"Note refinement failed: {e}")
            return {"tags": [], "links": [], "summary": "Błąd analizy AI."}

    def summarize_article(self, content: str) -> Optional[str]:
        """
        Summarizes an article using the NewsAgent system prompt.
        """
        if not self.available:
            return None

        system_prompt = ProjectConfig.PROMPTS.get('news_agent', {}).get('process_article_system_prompt', "Streszcz to.")

        try:
            # Truncate content if too long (sanity check, though models handle large context now)
            return self.generate_content(
                f"Podsumuj ten artykuł:\n\n{content[:50000]}",
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return None

    def generate_content_smart(
        self,
        prompt: str,
        task_type: str = "general",
        system_prompt: Optional[str] = None,
        format: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate content using SmartRouter for intelligent provider selection.

        Falls back to legacy generate_content if SmartRouter unavailable.

        Args:
            prompt: User prompt
            task_type: Task type for routing ("receipt_parsing", "rag_chat",
                      "summarization", "tagging", "ocr_vision", "general")
            system_prompt: Optional system instructions
            format: Output format ("json" for structured output)

        Returns:
            Generated content or None on failure
        """
        router = _get_smart_router()

        if router is None:
            # Fallback to legacy behavior
            return self.generate_content(prompt, system_prompt, format)

        try:
            from core.services.llm_router import TaskType

            # Map string to TaskType enum
            task_map = {
                "receipt_parsing": TaskType.RECEIPT_PARSING,
                "rag_chat": TaskType.RAG_CHAT,
                "rag_hyde": TaskType.RAG_HYDE,
                "summarization": TaskType.SUMMARIZATION,
                "tagging": TaskType.TAGGING,
                "ocr_vision": TaskType.OCR_VISION,
                "note_refinement": TaskType.NOTE_REFINEMENT,
                "general": TaskType.GENERAL,
            }

            task = task_map.get(task_type, TaskType.GENERAL)

            result = router.execute(
                task_type=task,
                prompt=prompt,
                system_prompt=system_prompt,
                format=format
            )

            if result.success:
                # Track cost
                try:
                    from utils.cost_tracker import get_tracker
                    tracker = get_tracker()
                    provider_name = result.provider_used.value
                    if "local" in provider_name:
                        provider_name = "local"
                    elif "openai" in provider_name:
                        provider_name = "openai"
                    elif "gemini" in provider_name:
                        provider_name = "gemini"
                    tracker.track_request(
                        provider=provider_name,
                        prompt=prompt,
                        response=result.content
                    )
                except Exception:
                    pass  # Cost tracking is optional

                return result.content
            else:
                logger.error(f"SmartRouter execution failed: {result.error}")
                return None

        except Exception as e:
            logger.error(f"SmartRouter error: {e}, falling back to legacy")
            return self.generate_content(prompt, system_prompt, format)

    # ========== ASYNC METHODS (Phase 1 - Receipt Pipeline Optimization) ==========

    async def generate_content_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Async version of generate_content() for non-blocking AI processing.

        Performance Benefits:
        - OpenAI: True async HTTP (1-2s non-blocking)
        - Gemini/Ollama: Thread executor (prevents blocking main thread)

        Usage:
            brain = UniversalBrain()
            result = await brain.generate_content_async(prompt)

        Args:
            prompt: User prompt text
            system_prompt: Optional system instructions
            format: Optional output format ("json")

        Returns:
            Generated text or None on failure
        """
        if not self.available:
            return None

        try:
            if self.provider == "openai":
                # OpenAI native async support
                return await self._openai_async(prompt, system_prompt, format)

            elif self.provider == "gemini":
                # Gemini SDK doesn't support async - use thread executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self._gemini_sync_call,
                    prompt,
                    system_prompt,
                    format
                )

            elif self.provider == "local":
                # Ollama doesn't support async - use thread executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self._ollama_sync_call,
                    prompt,
                    system_prompt,
                    format,
                    model_name
                )

        except Exception as e:
            logger.error(f"Async UniversalBrain ({self.provider}) Error: {e}")
            return None

    async def _openai_async(
        self,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str]
    ) -> Optional[str]:
        """
        OpenAI async implementation using AsyncOpenAI client.

        This is truly non-blocking HTTP - the event loop can handle
        other tasks while waiting for OpenAI response.
        """
        try:
            # Create async client
            async_client = openai.AsyncOpenAI(
                api_key=ProjectConfig.OPENAI_API_KEY
            )

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Async API call
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"} if format == "json" else None
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI async error: {e}")
            return None

    def _gemini_sync_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str]
    ) -> Optional[str]:
        """
        Gemini sync call (for use in thread executor).

        Since Gemini SDK doesn't support async natively, we run
        the sync version in a thread pool to avoid blocking.
        """
        try:
            model = self.model
            if system_prompt:
                model = genai.GenerativeModel(
                    model_name=ProjectConfig.GEMINI_MODEL,
                    system_instruction=system_prompt
                )

            generation_config = {}
            if format == "json":
                generation_config["response_mime_type"] = "application/json"

            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini sync call error: {e}")
            return None

    def _ollama_sync_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        format: Optional[str],
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Ollama sync call (for use in thread executor).

        Since Ollama SDK doesn't support async natively, we run
        the sync version in a thread pool.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            target_model = model_name or self.model_name
            response = ollama.chat(
                model=target_model,
                messages=messages,
                format='json' if format == 'json' else None
            )
            content = response['message']['content']

            # DeepSeek-R1 specific: remove <think> tags
            if "deepseek" in target_model.lower() and "<think>" in content:
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```', '', content).strip()

            return content

        except Exception as e:
            logger.error(f"Ollama sync call error: {e}")
            return None

# For backward compatibility
GeminiBrain = UniversalBrain