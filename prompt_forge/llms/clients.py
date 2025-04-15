# prompt_forge/llms/clients.py

import os
import time
import logging
import asyncio
import httpx # 使用 httpx 进行同步和异步请求 (Use httpx for sync/async requests)
from typing import Any, Dict, List, Optional, Union

# 导入基类 (Import base class)
from prompt_forge.core.base import BaseLLMClient

logger = logging.getLogger(__name__)

# 默认 OpenAI API 地址 (Default OpenAI API endpoint)
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
# 默认模型 (Default model)
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo" # Or newer models like gpt-4o-mini, gpt-4o

class OpenAIClient(BaseLLMClient):
    """
    用于与 OpenAI API (特别是 Chat Completions) 交互的 LLM 客户端。
    支持同步和异步操作，以及带并发控制的批量请求。

    LLM Client for interacting with the OpenAI API (specifically Chat Completions).
    Supports synchronous and asynchronous operations, and batch requests with concurrency control.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        api_base: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0, # Initial delay in seconds for retries
        max_concurrent_requests: int = 10, # Max concurrent requests for async batch
    ):
        """
        初始化 OpenAIClient。

        Args:
            api_key: OpenAI API 密钥。如果为 None，则尝试从环境变量 'OPENAI_API_KEY' 读取。
                         (OpenAI API key. If None, attempts to read from env var 'OPENAI_API_KEY'.)
            model: 要使用的 OpenAI 模型 ID。 (The OpenAI model ID to use.)
            api_base: (可选) API 的基础 URL，用于 Azure OpenAI 或代理。
                      (Optional base URL for the API, for Azure OpenAI or proxies.)
            timeout: 请求超时时间 (秒)。 (Request timeout in seconds.)
            max_retries: 发生可重试错误时的最大重试次数。
                         (Maximum number of retries on retryable errors.)
            retry_delay: 重试前的初始延迟时间 (秒)，会指数增长。
                         (Initial delay in seconds before retrying, increases exponentially.)
            max_concurrent_requests: 异步批量请求时的最大并发数。
                                     (Maximum concurrent requests for asynchronous batch processing.)
        """
        # ... (Initialization logic remains the same) ...
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided and 'OPENAI_API_KEY' environment variable not set."
            )

        self.api_key = api_key
        self.model = model
        self.api_base = api_base or DEFAULT_OPENAI_API_BASE
        self.chat_completions_url = f"{self.api_base.rstrip('/')}/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Semaphore for controlling concurrency in async batch requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Separate clients for sync and async operations
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

        logger.info(f"OpenAIClient initialized for model '{self.model}' at endpoint '{self.api_base}'. Max concurrent requests: {max_concurrent_requests}")


    # --- Client Initialization (Lazy) ---
    def _get_sync_client(self) -> httpx.Client:
        """Lazily initializes and returns the synchronous httpx client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.api_base,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazily initializes and returns the asynchronous httpx client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        """Returns the required HTTP headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_chat_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepares the payload for the Chat Completions API."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        payload.setdefault("temperature", 0.7)
        payload.setdefault("max_tokens", 1024)
        return payload

    # --- Core API Call Logic (with Retries) ---

    def _call_openai_api_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Makes a synchronous API call with retry logic."""
        # ... (Implementation remains the same) ...
        client = self._get_sync_client()
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(self.chat_completions_url, json=payload)
                response.raise_for_status() # Raise exception for 4xx/5xx errors
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exception = e
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    logger.error(f"Client error calling OpenAI API: {e.response.status_code} - {e.response.text}")
                    raise # Re-raise client errors immediately
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt+1}/{self.max_retries+1}): {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API call failed after {self.max_retries+1} attempts.")
                    raise ConnectionError(f"API call failed after retries: {last_exception}") from last_exception
        raise ConnectionError(f"API call failed unexpectedly: {last_exception}")


    async def _call_openai_api_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Makes an asynchronous API call with retry logic."""
        # ... (Implementation remains the same) ...
        client = self._get_async_client()
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                # Use semaphore to limit concurrency
                async with self.semaphore:
                    response = await client.post(self.chat_completions_url, json=payload)
                    response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exception = e
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    logger.error(f"Client error calling OpenAI API: {e.response.status_code} - {e.response.text}")
                    raise
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Async API call failed (attempt {attempt+1}/{self.max_retries+1}): {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Async API call failed after {self.max_retries+1} attempts.")
                    raise ConnectionError(f"Async API call failed after retries: {last_exception}") from last_exception
        raise ConnectionError(f"Async API call failed unexpectedly: {last_exception}")


    def _parse_response(self, response_data: Dict[str, Any]) -> str:
         """Extracts the generated text from the API response."""
         # ... (Implementation remains the same) ...
         try:
             if "choices" in response_data and response_data["choices"]:
                 first_choice = response_data["choices"][0]
                 if "message" in first_choice and "content" in first_choice["message"]:
                     return first_choice["message"]["content"].strip()
                 elif "text" in first_choice:
                     return first_choice["text"].strip()
             logger.warning(f"Could not extract content from OpenAI response: {response_data}")
             return ""
         except (KeyError, IndexError, TypeError) as e:
             logger.error(f"Error parsing OpenAI response structure: {e}. Response: {response_data}")
             return ""


    # --- BaseLLMClient Interface Implementation ---

    def generate(self, prompt: str, **kwargs) -> str:
        """同步生成单个响应。 (Synchronously generates a single response.)"""
        logger.debug(f"Generating sync completion for model '{self.model}'...")
        payload = self._prepare_chat_payload(prompt, **kwargs)
        try:
            response_data = self._call_openai_api_sync(payload)
            return self._parse_response(response_data)
        except Exception as e:
            logger.error(f"Sync generation failed: {e}", exc_info=True)
            raise

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """异步生成单个响应。 (Asynchronously generates a single response.)"""
        logger.debug(f"Generating async completion for model '{self.model}'...")
        payload = self._prepare_chat_payload(prompt, **kwargs)
        try:
            response_data = await self._call_openai_api_async(payload)
            return self._parse_response(response_data)
        except Exception as e:
            logger.error(f"Async generation failed: {e}", exc_info=True)
            raise

    # --- 修改 generate_batch ---
    # --- Modify generate_batch ---
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        同步生成批量响应 (通过串行调用 generate 实现)。
        Synchronously generates batch responses (by calling generate sequentially).
        """
        logger.debug(f"Generating sync batch of {len(prompts)} prompts for model '{self.model}' sequentially...")
        if not prompts:
            return []

        results = []
        for i, prompt in enumerate(prompts):
            try:
                # 调用同步的单个生成方法
                # Call the synchronous single generation method
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                # 在批处理中单个失败时记录错误，并添加空字符串或标记
                # Log error for single failure in batch and append empty string or marker
                logger.error(f"Error generating for prompt index {i} in sync batch: {e}")
                results.append("") # Append empty string on error

        return results

    async def agenerate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        异步生成批量响应，使用并发控制。
        Asynchronously generates batch responses with concurrency control.
        """
        # ... (Implementation remains the same) ...
        logger.debug(f"Generating async batch of {len(prompts)} prompts for model '{self.model}'...")
        if not prompts:
            return []

        tasks = []
        for prompt in prompts:
            payload = self._prepare_chat_payload(prompt, **kwargs)
            task = asyncio.create_task(self._call_openai_api_async(payload))
            tasks.append(task)

        results_data = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, res_data in enumerate(results_data):
            if isinstance(res_data, Exception):
                logger.error(f"Error in async batch generation for prompt index {i}: {res_data}")
                final_results.append("")
            elif isinstance(res_data, dict):
                final_results.append(self._parse_response(res_data))
            else:
                logger.error(f"Unexpected result type in async batch for prompt index {i}: {type(res_data)}")
                final_results.append("")

        return final_results


    # --- (Context manager methods remain the same) ---
    def close(self):
        """Closes the underlying synchronous HTTP client if it was initialized."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.info("Closed synchronous httpx client.")

    async def aclose(self):
        """Closes the underlying asynchronous HTTP client if it was initialized."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            logger.info("Closed asynchronous httpx client.")

    def __enter__(self):
        self._get_sync_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        self._get_async_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

