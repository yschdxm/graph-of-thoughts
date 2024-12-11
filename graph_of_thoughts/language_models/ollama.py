# ollama.py
import asyncio
import aiohttp
import backoff
import os
import random
import time
from typing import List, Dict, Union, Optional
from .abstract_language_model import AbstractLanguageModel

class Ollama(AbstractLanguageModel):
    """
    The Ollama class handles interactions with Ollama models using the provided configuration.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "ollama", cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.base_url: str = self.config.get("base_url", "http://localhost:11434")
        self.temperature: float = self.config.get("temperature", 0.7)
        self.max_tokens: int = self.config.get("max_tokens", 512)
        self.stop: Union[str, List[str]] = self.config.get("stop", [])
        self.session: Optional[aiohttp.ClientSession] = None
        self._loop = None

    async def initialize_session(self):
        """Initialize aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[Dict], Dict]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._async_query(query, num_responses))

    async def _async_query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[Dict], Dict]:
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = await self.chat([{"role": "user", "content": query}])
        else:
            await self.initialize_session()
            tasks = [self.chat([{"role": "user", "content": query}]) for _ in range(num_responses)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            response = []
            for res in responses:
                if not isinstance(res, Exception):
                    response.append(res)
            
            # Retry failed requests
            retry_count = num_responses - len(response)
            if retry_count > 0:
                retry_tasks = [self.chat([{"role": "user", "content": query}]) for _ in range(retry_count)]
                retry_responses = await asyncio.gather(*retry_tasks, return_exceptions=True)
                response.extend([r for r in retry_responses if not isinstance(r, Exception)])

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=10, max_tries=6)
    async def chat(self, messages: List[Dict]) -> Dict:
        await self.initialize_session()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": self.stop
            },
            "stream": False
        }
        
        start_time = time.time()
        request_id = id(self)
        self.logger.info(f"Request {request_id} started at {start_time}")
        
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_msg = await resp.text()
                raise RuntimeError(f"API request failed with status {resp.status}: {error_msg}")
            
            data = await resp.json()
            self.logger.info(
                f"Response from Ollama: {data}"
                f"\nRequest {request_id} finished at {time.time()}"
                f"\nTime elapsed: {time.time() - start_time} seconds"
            )
            return data

    async def close_async(self):
        """Asynchronously close session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def close(self):
        """Synchronously close session"""
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(self.close_async())
                else:
                    new_loop = asyncio.new_event_loop()
                    new_loop.run_until_complete(self.close_async())
                    new_loop.close()
            except Exception:
                pass
            self.session = None

    def get_response_texts(
        self, query_response: Union[List[Dict], Dict]
    ) -> List[str]:
        if not isinstance(query_response, list):
            query_response = [query_response]
            
        texts = []
        for response in query_response:
            if "message" in response and "content" in response["message"]:
                texts.append(response["message"]["content"])
        return texts

    def __del__(self):
        """Destructor - log if session not closed"""
        try:
            if self.session and not self.session.closed:
                self.logger.warning("Session was not properly closed")
        except Exception:
            pass
