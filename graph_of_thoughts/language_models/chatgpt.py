# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
import asyncio
import aiohttp


import backoff
import os
import random
import time
from typing import List, Dict, Union, Optional
from openai import OpenAI, OpenAIError
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
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The account organization is the organization that is used for chatgpt.
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
        self.base_url: str = self.config["base_url"]
        # Initialize the OpenAI Client
        self.client = OpenAI(api_key=self.api_key, organization=self.organization, base_url=self.base_url)
        self.session: Optional[aiohttp.ClientSession] = None

        # try:
        #     self.loop = asyncio.get_event_loop()
        # except RuntimeError:
        #     self.loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(self.loop)

        self._loop = None  # 不再主动创建事件循环

    async def initialize_session(self):
        """Initialize aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Organization": self.organization,
                    "Content-Type": "application/json"
                }
            )
            
    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
        :rtype: Dict
        """
        # return self.loop.run_until_complete(self._async_query(query, num_responses))
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
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = await self.chat([{"role": "user", "content": query}], num_responses)
        else:
            await self.initialize_session()
            tasks = [self.chat([{"role": "user", "content": query}]) for _ in range(num_responses)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            response = []
            # next_try = num_responses
            # total_num_attempts = num_responses
            # while num_responses > 0 and total_num_attempts > 0:
            #     try:
            #         assert next_try > 0
            #         for _ in range(next_try):
            #             res = self.chat([{"role": "user", "content": query}])
            #             response.append(res)

            #         response.append(res)
            #         num_responses -= next_try
            #         next_try = min(num_responses, next_try)
            #     except Exception as e:
            #         next_try = (next_try + 1) // 2
            #         self.logger.warning(
            #             f"Error in chatgpt: {e}, trying again with {next_try} samples"
            #         )
            #         time.sleep(random.randint(1, 3))
            #         total_num_attempts -= 1
            retry_tasks = []
            for i, res in enumerate(responses):
                if isinstance(res, Exception):
                    self.logger.warning(f"Request failed: {res}, adding to retry queue")
                    retry_tasks.append(self.chat([{"role": "user", "content": query}]))
                else:
                    response.append(res)
            
            if retry_tasks:
                retry_responses = await asyncio.gather(*retry_tasks, return_exceptions=True)
                response.extend([r for r in retry_responses if not isinstance(r, Exception)])

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        start_time = time.time()
        request_id = id(self)
        self.logger.info(f"Request {request_id} started at {start_time}")
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        # response = self.client.chat.completions.create(
        #     model=self.model_id,
        #     messages=messages,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     n=num_responses,
        #     stop=self.stop,
        # )

        # self.prompt_tokens += response.usage.prompt_tokens
        # self.completion_tokens += response.usage.completion_tokens
        # prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        # completion_tokens_k = float(self.completion_tokens) / 1000.0
        # self.cost = (
        #     self.prompt_token_cost * prompt_tokens_k
        #     + self.response_token_cost * completion_tokens_k
        # )
        # self.logger.info(
        #     f"This is the response from chatgpt: {response}"
        #     f"\nThis is the cost of the response: {self.cost}"
        # )
        # return response
        await self.initialize_session()
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stop": self.stop,
        }

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_msg = await resp.text()
                raise OpenAIError(f"API request failed with status {resp.status}: {error_msg}")
            
            data = await resp.json()
            response = ChatCompletion(**data)
            
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )
            self.logger.info(
                f"Response from chatgpt: {response}"
                f"\nCost: {self.cost}"
                f"\nRequest {request_id} finished at {time.time()}"
                f"\nTime elapsed: {time.time() - start_time} seconds"
            )
            return response

    async def close(self):
        """异步关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    # def close(self):
    #     """同步关闭方法"""
    #     if self.session and not self.session.closed:
    #         self.loop.run_until_complete(self.session.close())
    #         self.loop.close()
    def close(self):
        """同步关闭方法"""
        if self.session and not self.session.closed:
            # 安全地关闭会话
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(self.close_async())
                else:
                    # 如果事件循环已关闭，创建临时循环
                    new_loop = asyncio.new_event_loop()
                    new_loop.run_until_complete(self.close_async())
                    new_loop.close()
            except Exception:
                pass
            self.session = None

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]

    # def __del__(self):
    #     """新增析构方法"""
    #     if self.session and not self.session.closed:
    #         asyncio.get_event_loop().run_until_complete(self.close())
    # def __del__(self):
    #     """同步析构方法"""
    #     self.close()
    def __del__(self):
        """析构方法 - 仅记录不执行关闭操作"""
        try:
            if self.session and not self.session.closed:
                self.logger.warning("Session was not properly closed")
        except Exception:
            pass