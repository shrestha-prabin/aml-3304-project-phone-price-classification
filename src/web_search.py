import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def web_search(data):
    prompt = """Given phone specs: screen size {screen_size} inches, RAM {ram} MB, 
            storage {int_memory} GB, battery {battery_power} mAh, main camera {pc} MP, at least talk time {talk_time} hrs, generate a JSON list of 4 phone models.
            Each entry should include brand, name, image_url, price (in CAD $), and 
            specs (must include screen_size, ram, int_memory, battery_power, pc, talk_time) as a JSON object.
            The phones should be real models available in the market."""

    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=prompt.format(
            screen_size=data["screen_size"],
            ram=data["ram"],
            int_memory=data["int_memory"],
            battery_power=data["battery_power"],
            pc=data["pc"],
            talk_time=data["talk_time"],
        ),
    )
    # Extract the json content from the response
    data = re.findall(r"```json(.*?)```", response.output_text, re.DOTALL)
    data_json = json.loads(data[0])

    return data_json
