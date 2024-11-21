"""This module contains the main logic for the brain of JARVIS served by OpenAI Chat-GPT model.

Returns:
    _type_: _description_
"""
import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
import time
class JARVIS:
    API_REQUEST_LIMIT = 10
    API_REQUESTS_PM = 0
    DEFAULT_SYSTEM_PROMPT : Dict = {
    "role": "system",
    "content": (
        "You are J.A.R.V.I.S., a sophisticated AI assistant from the Iron Man universe. "
        "You respond in a polite, articulate, and formal tone, offering intelligent insights "
        "and practical assistance while maintaining a composed and witty demeanor."
    )
}
    
    def __init__(self) -> None:
        #load .env variables
        load_dotenv()
        self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.start_time = time.time()
    def reset_requests(self):
        """Resets the request count after a minute has passed."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            JARVIS.API_REQUESTS_PM = 0
            self.start_time = current_time

    def send_prompt(self, messages: List[Dict[str,str]],model :str = "gpt-4o-mini") -> None:
        """Sends messages to the OpenAi model based on specification.

        Args:
            messages (List[Dict[str,str]]): _description_
            model (str, optional): _description_. Defaults to "gpt-4o-mini".

        Returns:
            _type_: _description_
        """
        #Check whether to reset requests per minute
        self.reset_requests()
        
        
        JARVIS.API_REQUESTS_PM += 1
        if JARVIS.API_REQUESTS_PM <= JARVIS.API_REQUEST_LIMIT:
            request = self.open_ai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            return request.choices[0].message.content.strip()
        else:
            return f"""Limit of {JARVIS.API_REQUEST_LIMIT}
        requests per minute has been reached."""


