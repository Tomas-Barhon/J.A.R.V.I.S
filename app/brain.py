"""This module contains the main logic for the brain of JARVIS served by OpenAI Chat-GPT model.

Returns:
    _type_: _description_
"""
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
import time
class JARVIS:
    API_REQUEST_LIMIT = 10
    API_REQUESTS_PM = 0
    DEFAULT_SYSTEM_PROMPT : Tuple = (
            "system",
            "You are J.A.R.V.I.S., a sophisticated AI assistant from the Iron Man universe. "
        "You respond in a polite, articulate, and formal tone, offering intelligent insights "
        "and practical assistance while maintaining a composed and witty demeanor.",
        )
    
    def __init__(self, model :str = "gpt-3.5-turbo-1106") -> None:
        #load .env variables
        load_dotenv()
        self.open_ai_client = ChatOpenAI(model=model,
                                         api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

        self.langchain_graph = StateGraph(state_schema=MessagesState)
        self.langchain_graph.add_edge(START, "model")
        self.langchain_graph.add_node("model", self.call_model)

        # Add memory
        self.chat_memory = MemorySaver()
        self.app = self.langchain_graph.compile(checkpointer=self.chat_memory)
        
        self.prompt = ChatPromptTemplate.from_messages(
    [
        JARVIS.DEFAULT_SYSTEM_PROMPT,
        MessagesPlaceholder(variable_name="messages"),
    ]
)
        self.chain = self.prompt | self.open_ai_client
        self.agent = create_react_agent(self.chain, tools=self.get_tools())
        self.start_time = time.time()

    def get_tools(self):
        ...

    def reset_requests(self):
        """Resets the request count after a minute has passed."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            JARVIS.API_REQUESTS_PM = 0
            self.start_time = current_time
    
    def call_model(self, state: MessagesState):
        response = self.chain.invoke(state["messages"])
        # Update message history with response:
        return {"messages": response}

    def send_prompt(self, messages: str, config: dict) -> None:
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
            input_messages = [HumanMessage(messages)]
            output = self.app.invoke({"messages": input_messages}, config)
            state = self.app.get_state(config).values

            for message in state["messages"]:
                message.pretty_print()
            return output["messages"][-1].content.strip()
        else:
            return f"""Limit of {JARVIS.API_REQUEST_LIMIT}
        requests per minute has been reached."""
