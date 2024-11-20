"""This module contains the main logic for the brain of JARVIS served by OpenAI Chat-GPT model.

Returns:
    _type_: _description_
"""
import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

class JARVIS:
    DEFAULT_SYSTEM_PROMPT : Dict = {"role": "system", "content": "You are a helpful assistant."}
    
    def __init__(self) -> None:
        #load .env variables
        load_dotenv()
        self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_prompt(self, messages: List[Dict[str,str]],model :str = "gpt-4o-mini") -> None:
        """Sends messages to the OpenAi model based on specification.

        Args:
            messages (List[Dict[str,str]]): _description_
            model (str, optional): _description_. Defaults to "gpt-4o-mini".

        Returns:
            _type_: _description_
        """
        request = self.open_ai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return request.choices[0].message.content.strip()





