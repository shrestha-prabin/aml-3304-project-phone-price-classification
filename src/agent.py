import json
import os
import re
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()


# The original question or task
class State(TypedDict):
    """State class to track agent's data throughout the workflow"""

    data: any
    content: any


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def content_generation_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Given phone specs: screen size {screen_size} inches, RAM {ram} MB, 
            storage {int_memory} GB, battery {battery_power} mAh, main camera {pc} MP, at least talk time {talk_time} hrs, generate a JSON list of 4 phone models.
            Each entry should include brand, name, image_url, price, and 
            specs (must include screen_size, ram, int_memory, battery_power, pc, talk_time) as a JSON object.
            The phones should be real models available in the market.""",
    )

    # Format the prompt with the input text from the state
    message = HumanMessage(
        content=prompt.format(
            screen_size=state["data"]["screen_size"],
            ram=state["data"]["ram"],
            int_memory=state["data"]["int_memory"],
            battery_power=state["data"]["battery_power"],
            pc=state["data"]["pc"],
            talk_time=state["data"]["talk_time"],
        )
    )

    # Invoke the language model to generate the text based on the prompt
    text = llm.invoke([message]).content.strip()

    # Extract the json content from the response
    data = re.findall(r"```json(.*?)```", text, re.DOTALL)
    data_json = json.loads(data[0])

    return {"content": data_json}


def build_agent():
    workflow = StateGraph(State)

    # Add nodes to the graph
    workflow.add_node("content_generation_node", content_generation_node)

    # Add edges to the graph
    workflow.set_entry_point(
        "content_generation_node"
    )  # Set the entry point of the graph
    workflow.add_edge("content_generation_node", END)

    # Compile the graph
    app = workflow.compile()

    return app


def run_agent(data):
    app = build_agent()

    print("Running agent with data:", data)
    # Create the initial state
    state_input = {"data": data}

    # Run the agent's full workflow on our sample text
    result = app.invoke(state_input)

    # Print each component of the result:
    print("Content:", result["content"])

    return result["content"]
