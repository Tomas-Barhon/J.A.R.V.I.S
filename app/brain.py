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
from toolkit import Toolset
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langgraph.prebuilt.tool_executor import ToolExecutor
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

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
                                         api_key=os.getenv("OPENAI_API_KEY")
                                        ,temperature=0, streaming=False)
        
        #toolkit for the agent
        self.toolkit = Toolset()
        
        self.agent = self.create_agent(client=self.open_ai_client
                                       , tools=self.toolkit.get_tools(),
                                       system_prompt=JARVIS.DEFAULT_SYSTEM_PROMPT)


        # Building LangGraph workflow
        self.langchain_graph = StateGraph(state_schema=AgentState)
        
        self.langchain_graph.add_node("model", self.call_model)
        self.langchain_graph.add_node("action", self.execute_tools)
        self.langchain_graph.set_entry_point("model")
        self.langchain_graph.add_conditional_edges(
            "model", self.should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        self.langchain_graph.add_edge("action", "model")
        
    

        # Add memory
        self.chat_memory = MemorySaver()
        self.app = self.langchain_graph.compile()

        

        self.start_time = time.time()


    def reset_requests(self):
        """Resets the request count after a minute has passed."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            JARVIS.API_REQUESTS_PM = 0
            self.start_time = current_time

    def create_agent(self, client: ChatOpenAI, tools: list, system_prompt: Tuple):
        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(client,
                        tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools,
                                    max_iterations=2, verbose=True, return_intermediate_steps=False
                                    )
        return agent
        
    def call_model(self, state: AgentState):
        print("Calling model")
        
        outcome = self.agent.invoke(state)
        # Update message history with response:
        return {"agent_outcome": outcome}

    def execute_tools(self, state: AgentState):
        agent_action = state["agent_outcome"]
        # Execute the tool
        tool_executor = ToolExecutor(self.toolkit.get_tools())
        output = tool_executor.invoke(agent_action)
        print(f"The agent action is {agent_action}")
        print(f"The tool result is: {output}")
        # Return the output
        return {"intermediate_steps": [(agent_action, str(output))]}

    def should_continue(self, state: AgentState):
        if isinstance(state["agent_outcome"], AgentFinish):
            return "end"
        else:
            return "continue"

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
            state = {
            "input": messages,
            "chat_history": []}
            #output = self.app.invoke(state)
            for s in self.app.stream(state):
                print(list(s.values())[0])
                print("----")
            print(output)
            for message in state["messages"]:
                message.pretty_print()
            return output["messages"][-1].content.strip()
        else:
            return f"""Limit of {JARVIS.API_REQUEST_LIMIT}
        requests per minute has been reached."""
