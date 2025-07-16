# ollama.py
import os
import time
import random
import requests
import asyncio
import aiohttp
import backoff
from typing import List, Dict, Union, Any
from .abstract_language_model import AbstractLanguageModel

class Ollama(AbstractLanguageModel):
    """
    The Ollama class handles interactions with Ollama models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "ollama", cache: bool = False
    ) -> None:
        """
        Initialize the Ollama instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'ollama'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.base_url: str = self.config.get("base_url", "http://localhost:11434")
        self.temperature: float = self.config.get("temperature", 0.7)
        self.max_tokens: int = self.config.get("max_tokens", 512)
        self.stop: Union[str, List[str]] = self.config.get("stop", None)
        self.max_concurrent: int = self.config.get("max_concurrent", 5)  # Max concurrent requests
        
        # Ollama doesn't have token costs since it's local
        self.prompt_token_cost: float = 0.0
        self.response_token_cost: float = 0.0
        
        # Initialize token counters
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[Dict], Dict]:
        """
        Query the Ollama model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the Ollama model.
        :rtype: Union[List[Dict], Dict]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        messages = [{"role": "user", "content": query}]
        
        # Use async for multiple responses, sync for single response
        if num_responses > 1:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                self.async_chat(messages, num_responses)
            )
        else:
            response = self.chat(messages, num_responses)

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> Union[List[Dict], Dict]:
        """
        Send chat messages to the Ollama model and retrieves the model's response.
        Implements backoff on request errors.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The Ollama model's response(s).
        :rtype: Union[List[Dict], Dict]
        """
        # For single response, use synchronous method
        if num_responses == 1:
            response = self._chat_completion(messages)
            self._update_token_usage([response])
            return response
        else:
            # For multiple responses, use async method
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.async_chat(messages, num_responses)
            )

    async def async_chat(
        self, messages: List[Dict], num_responses: int
    ) -> List[Dict]:
        """
        Asynchronously send chat messages to the Ollama model and retrieves multiple responses.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses.
        :type num_responses: int
        :return: List of responses from the Ollama model.
        :rtype: List[Dict]
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(num_responses):
                task = asyncio.ensure_future(
                    self._async_chat_completion(session, semaphore, messages)
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            self._update_token_usage(responses)
            return responses

    async def _async_chat_completion(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, messages: List[Dict]
    ) -> Dict:
        """Internal method for async API calls"""
        async with semaphore:
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            if self.stop:
                payload["stop"] = self.stop
                
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _chat_completion(self, messages: List[Dict]) -> Dict:
        """Internal method for synchronous API calls"""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if self.stop:
            payload["stop"] = self.stop
            
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def _update_token_usage(self, responses: List[Dict]) -> None:
        """Update token usage statistics from Ollama responses"""
        for response in responses:
            self.prompt_tokens += response.get("prompt_eval_count", 0)
            self.completion_tokens += response.get("eval_count", 0)
        # Ollama is free so cost remains 0

    def get_response_texts(
        self, query_response: Union[List[Dict], Dict]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from Ollama.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
            
        return [
            response["message"]["content"]
            for response in query_response
        ]

    # The following methods are maintained for compatibility but are not functional with Ollama
    # They are marked as "INVALID FOR OLLAMA" in the docstrings

    def create_model(self, source_model: str, new_model: str) -> None:
        """
        INVALID FOR OLLAMA: Create a new model from an existing one.
        Ollama requires different parameters for model creation.
        """
        raise NotImplementedError("Ollama requires different parameters for model creation")

    def copy_model(self, source_model: str, new_model: str) -> None:
        """
        INVALID FOR OLLAMA: Copy a model.
        Use the /api/copy endpoint instead.
        """
        raise NotImplementedError("Use the /api/copy endpoint for Ollama")

    def delete_model(self, model_name: str) -> None:
        """
        INVALID FOR OLLAMA: Delete a model.
        Use the /api/delete endpoint instead.
        """
        raise NotImplementedError("Use the /api/delete endpoint for Ollama")

    def pull_model(self, model_name: str) -> None:
        """
        INVALID FOR OLLAMA: Pull a model from the registry.
        Use the /api/pull endpoint instead.
        """
        raise NotImplementedError("Use the /api/pull endpoint for Ollama")

    def push_model(self, model_name: str) -> None:
        """
        INVALID FOR OLLAMA: Push a model to the registry.
        Use the /api/push endpoint instead.
        """
        raise NotImplementedError("Use the /api/push endpoint for Ollama")

    def generate_embeddings(self, input_text: Union[str, List[str]]) -> List[List[float]]:
        """
        INVALID FOR OLLAMA: Generate embeddings from text.
        Use the /api/embeddings endpoint instead.
        """
        raise NotImplementedError("Use the /api/embeddings endpoint for Ollama")
