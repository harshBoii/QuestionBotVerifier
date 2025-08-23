import os
import uuid
from typing import TypedDict, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- Setup Environment ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- Define Agent State ---

class EmailAgentState(TypedDict):
    """
    Represents the state of our email generation agent.
    
    Attributes:
        prompt: The core message or purpose of the email.
        company_type: The industry or type of the company (e.g., 'web dev agency', 'art gallery').
        email_type: The category of the email (e.g., 'welcome', 'project update', 'promotional').
        session_id: A unique identifier for the session.
        email_vibe: The determined tone and style for the email.
        generated_email: The final, formatted email content.
    """
    prompt: str
    company_type: str
    email_type: str
    session_id: str
    email_vibe: str
    generated_email: str
    company_name:str

# --- Define Graph Nodes ---

def determine_vibe_node(state: EmailAgentState) -> EmailAgentState:
    """
    Determines the appropriate "vibe" (tone, style, voice) for the email
    based on the company and email type.
    """
    print("---NODE: DETERMINING EMAIL VIBE---")
    company_type = state["company_type"]
    email_type = state["email_type"]

    prompt = f"""
    Based on the company type and the purpose of the email, describe the ideal vibe and tone.
    The description should be a few keywords or a short phrase that can guide a copywriter.

    Company Type: '{company_type}'
    Email Type: '{email_type}'

    Example 1:
    - Company Type: 'Law Firm'
    - Email Type: 'First client meeting confirmation'
    - Vibe: Professional, formal, clear, reassuring, precise.

    Example 2:
    - Company Type: 'Web Dev Agency'
    - Email Type: 'Welcome Email'
    - Vibe: Techie, modern, enthusiastic, friendly, maybe slightly informal with an emoji.

    Example 3:
    - Company Type: 'Painter / Artist'
    - Email Type: 'New exhibition announcement'
    - Vibe: Artistic, creative, descriptive, evocative, personal.

    Now, determine the vibe for the following:
    - Company Type: '{company_type}'
    - Email Type: '{email_type}'
    - Vibe:
    """

    response = llm.invoke(prompt)
    vibe = response.content.strip()
    print(f"Determined Vibe: {vibe}")

    state["email_vibe"] = vibe
    return state
def generate_email_node(state: EmailAgentState) -> EmailAgentState:
    """
    Generates the final email with a subject and HTML body, separated by '|||---|||'.
    """
    print("---NODE: GENERATING EMAIL---")
    user_prompt = state["prompt"]
    email_vibe = state["email_vibe"]
    email_type = state["email_type"]
    company_type = state["company_type"]
    company_name = state["company_name"]

    # Updated prompt to request HTML output and a specific separator
    prompt = f"""
    You are an expert email copywriter for a '{company_type}' company Whose name is `{company_name}`.
    Your task is to write a complete '{email_type}' email based on the following instructions.

    **Vibe to adopt:** {email_vibe}
    **Core Message:** {user_prompt}

    **Instructions:**
    1.  Write a concise and compelling subject line.
    2.  Write the email body using standard HTML tags (`<p>`, `<strong>`, `<em>`, `<ul>`, `<li>`, `<a>`, etc.). Do NOT include `<html>`, `<head>`, or `<body>` tags.
    3.  Format your entire response as follows:
        [Your Subject Line Here]|||---|||[Your HTML Body Here]

    **Example Output:**
    A Special Welcome Offer Just For You!|||---|||<p>Hi there,</p><p>Welcome to our community! We're thrilled to have you. As a special thank you, here is a <strong>10% discount</strong> on your next purchase.</p><p>Best,<br>The Team</p>

    Now, generate the email based on the provided details.
    """

    response = llm.invoke(prompt)
    generated_email = response.content.strip()
    print("---Generated Raw Output---")
    print(generated_email)

    state["generated_email"] = generated_email
    return state


def save_email_node(state: EmailAgentState):
    """
    Saves the generated email to a markdown file.
    """
    print("---NODE: SAVING EMAIL TO FILE---")
    session_id = state["session_id"]
    email_content = state["generated_email"]
    
    # Ensure the directory exists
    output_dir = "generated_emails"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/email_session_{session_id}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Email for session {session_id}\n\n")
        f.write(f"**Company Type:** {state['company_type']}\n")
        f.write(f"**Email Type:** {state['email_type']}\n")
        f.write(f"**Prompt:** {state['prompt']}\n\n")
        f.write("---\n\n")
        f.write(email_content)
            
    print(f"✅ Email saved successfully to {filename}")
    return state

# --- Build the Graph ---

workflow = StateGraph(EmailAgentState)

# Add nodes to the graph
workflow.add_node("determine_vibe", determine_vibe_node)
workflow.add_node("generate_email", generate_email_node)
workflow.add_node("save_email", save_email_node)

# Set the entry point
workflow.set_entry_point("determine_vibe")

# Add edges to define the flow
workflow.add_edge("determine_vibe", "generate_email")
workflow.add_edge("generate_email", "save_email")
workflow.add_edge("save_email", END)

# Compile the graph into a runnable app
email_agent_app = workflow.compile()

