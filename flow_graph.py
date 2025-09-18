import os
import json
from dotenv import load_dotenv
from typing import TypedDict
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from template import flow_prompt  # your ChatPromptTemplate

load_dotenv()

# --- Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# --- State Definition ---
class FlowState(TypedDict):
    description: str
    output: str  # we’ll store the JSON string here
    prompt:str

# --- Node functions ---
async def apply_prompt(state: FlowState) -> dict:
    """Format description into a prompt."""
    prompt_value = flow_prompt.format(description=state["description"])
    return {"description": state["description"], "prompt": prompt_value}

async def call_llm(state: FlowState) -> dict:
    """Call Gemini with the formatted prompt."""
    response = await llm.ainvoke(state["prompt"])
    return {"description": state["description"], "output": response.content}

# --- Graph setup ---
graph = StateGraph(FlowState)

graph.add_node("format_prompt", apply_prompt)
graph.add_node("llm_call", call_llm)

# wiring: entry → format_prompt → llm_call → END
graph.set_entry_point("format_prompt")
graph.add_edge("format_prompt", "llm_call")
graph.add_edge("llm_call", END)

compiled_graph = graph.compile()

# --- Runner function ---
# async def generate_flow_from_description(description: str) -> dict:
#     if not description:
#         return {"error": "Missing description"}

#     result = await compiled_graph.ainvoke({"description": description})

#     response_text = result.get("output", "")

#     # Try to parse JSON safely
#     try:
#         return json.loads(response_text)
#     except Exception:
#         return {"raw": response_text}
def safe_json_parse(text: str):
    # Remove markdown fences like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # If it’s still wrapped, try extracting with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {"raw": text}


async def generate_flow_from_description(description: str) -> dict:
    if not description:
        return {"error": "Missing description"}

    result = await compiled_graph.ainvoke({"description": description})

    response_text = result.get("output", "") if isinstance(result, dict) else str(result)

    return safe_json_parse(response_text)
