"""This module contains the main logic for the brain of JARVIS served by OpenAI Chat-GPT model.

Returns:
    _type_: _description_
"""
import os
import operator
import time
from langchain import hub
from typing import Tuple, TypedDict, Annotated, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from src.tools.toolkit import Toolset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolNode, tools_condition

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

class JARVIS:
    API_REQUEST_LIMIT = 10
    API_REQUESTS_PM = 0
    DEFAULT_SYSTEM_PROMPT: Tuple = (
        "system",
        "You are J.A.R.V.I.S., a sophisticated AI assistant from the Iron Man universe. "
        "You respond in a polite, articulate, and formal tone, offering intelligent insights "
        "and practical assistance while maintaining a composed and witty demeanor.",
    )

    def __init__(self, model: str = "gpt-3.5-turbo-1106") -> None:
        # load .env variables
        load_dotenv()
        self.llm = init_chat_model(model=model, 
                        api_key=os.getenv("OPENAI_API_KEY"), 
                        temperature=0.2, streaming=False)
        self.toolkit = Toolset()

        self.agent = self.llm.bind_tools(self.toolkit.get_tools())
        # Building LangGraph workflow
        self.langchain_graph_builder = StateGraph(state_schema=AgentState)
        self.langchain_graph_builder.add_node("model", self.call_model)
        
        tool_node = ToolNode(tools=self.toolkit.get_tools())
        self.langchain_graph_builder.add_node("tool_node", tool_node)
        self.langchain_graph_builder.add_conditional_edges("model", tools_condition)
        self.langchain_graph_builder.add_edge("tool_node", "model")
        self.langchain_graph_builder.set_entry_point("model")

        # Add memory
        # TODO: Upgrade to PostgreSQL database
        self.chat_memory = MemorySaver()
        self.graph = self.langchain_graph_builder.compile(checkpointer=self.chat_memory)
        self.start_time = time.time()

    def reset_requests(self):
        """Resets the request count after a minute has passed."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            JARVIS.API_REQUESTS_PM = 0
            self.start_time = current_time

    def call_model(self, state: AgentState):
        print("Calling model")
        # Update message history with response:
        return {"messages": [self.llm.invoke(state["messages"])]}

    def send_prompt(self, messages: str, config: dict) -> None:
        """Sends messages to the OpenAi model based on specification.

        Args:
            messages (List[Dict[str,str]]): _description_
            model (str, optional): _description_. Defaults to "gpt-4o-mini".

        Returns:
            _type_: _description_
        """
        # Check whether to reset requests per minute
        self.reset_requests()

        JARVIS.API_REQUESTS_PM += 1
        if JARVIS.API_REQUESTS_PM <= JARVIS.API_REQUEST_LIMIT:
            events = self.graph.stream(
            {"messages": [{"role": "user", "content": messages}]},
            config,
            stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()
        else:
            return f"""Limit of {JARVIS.API_REQUEST_LIMIT}
        requests per minute has been reached."""
