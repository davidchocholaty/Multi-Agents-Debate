import backoff
import time
import random
from openai.error import RateLimitError, APIError, APIConnectionError
from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context

import os
from dotenv import load_dotenv
import openai
load_dotenv()

openai.api_type = "azure"
openai.api_base = os.environ['OPEN_AI_API_BASE']
openai.api_version = os.environ['OPEN_AI_API_VERSION']
openai.api_key = os.environ['OPEN_AI_API_KEY']

support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0) -> None:
        """Create an agent

        Args:
            model_name(str): model name
            name (str): name of this agent
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=20)
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        """make a query

        Args:
            messages (list[dict]): chat history in turbo format
            max_tokens (int): max token in api call
            api_key (str): openai api key
            temperature (float): sampling temperature

        Raises:
            OutOfQuotaException: the apikey has out of quota
            AccessTerminatedException: the apikey has been ban

        Returns:
            str: the return msg
        """
        time.sleep(self.sleep_time)
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
        try:
            if self.model_name in support_models:
                response = openai.ChatCompletion.create(
                    # engine="gpt-35-turbo",
                    engine="gpt-4o",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                '''
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )'''
                gen = response['choices'][0]['message']['content']
            return gen

        except RateLimitError as e:
            if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
                raise OutOfQuotaException(api_key)
            elif "Your access was terminated due to violation of our policies" in e.user_message:
                raise AccessTerminatedException(api_key)
            else:
                raise e

    def set_meta_prompt(self, meta_prompt: str):
        """Set the meta_prompt

        Args:
            meta_prompt (str): the meta prompt
        """
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        """Add an new event in the memory

        Args:
            event (str): string that describe the event.
        """
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float=None):
        """Query for answer

        Args:
        """
        # query
        num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        max_token = model2max_context[self.model_name] - num_context_token
        return self.query(self.memory_lst, max_token, api_key=self.openai_api_key, temperature=temperature if temperature else self.temperature)

