# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import os
import random
import time
import asyncio
import aiohttp
from typing import List, Dict, Union, Any
from openai import OpenAI, AsyncOpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        self.temperature: float = self.config["temperature"]
        self.max_tokens: int = self.config["max_tokens"]
        self.stop: Union[str, List[str]] = self.config["stop"]
        self.organization: str = self.config["organization"]
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        self.base_url: str = self.config.get("base_url", None)
        
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
            
        # Initialize clients
        client_kwargs = {
            "api_key": self.api_key,
            "organization": self.organization,
            "timeout": 3600,
            "max_retries": 1000,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        self.max_concurrent: int = 5  # Max concurrent async requests

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[Dict], Dict]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
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
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on request errors.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response(s).
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
        Asynchronously send chat messages to the OpenAI model and retrieves multiple responses.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses.
        :type num_responses: int
        :return: List of responses from the OpenAI model.
        :rtype: List[Dict]
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        tasks = []
        for _ in range(num_responses):
            task = asyncio.ensure_future(
                self._async_chat_completion(semaphore, messages)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        self._update_token_usage(responses)
        return responses

    async def _async_chat_completion(
        self, semaphore: asyncio.Semaphore, messages: List[Dict]
    ) -> Dict:
        """Internal method for async API calls"""
        async with semaphore:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,  # We handle multiple responses at a higher level
                    stop=self.stop,
                )
                return response
            except OpenAIError as e:
                self.logger.error(f"Error in async_chat_completion: {e}")
                raise e

    def _chat_completion(self, messages: List[Dict]) -> Dict:
        """Internal method for synchronous API calls"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stop=self.stop,
            )
            return response
        except OpenAIError as e:
            self.logger.error(f"Error in chat_completion: {e}")
            raise e

    def _update_token_usage(self, responses: List[Dict]) -> None:
        """Update token usage statistics from responses"""
        for response in responses:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            prompt_tokens_k = float(response.usage.prompt_tokens) / 1000.0
            completion_tokens_k = float(response.usage.completion_tokens) / 1000.0
            self.cost += (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )

    def get_response_texts(
        self, query_response: Union[List[Dict], Dict]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from OpenAI.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
            
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
