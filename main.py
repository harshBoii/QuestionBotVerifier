# main.py

import uuid
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import uvicorn
from langgraph.types import Command
# Import the agent_app from your corrected agent.py
from agent import agent_app
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import PlainTextResponse
from emailAgent import email_agent_app, EmailAgentState

# Define the data models for the request bodies
class FeedbackContext(BaseModel):
    profession: str
    work_experience: str
    name: str

class ContinueBody(BaseModel):
    session_id: str
    answer: str
    # The client must now send back the checkpoint_id
    checkpoint_id: str

class EmailRequest(BaseModel):
    company_name:str
    company_type: str
    email_type: str
    prompt: str

# Create the FastAPI app instance
app = FastAPI(
    title="Interactive Feedback Chatbot API",
    description="An API to have a real conversation with the LangGraph agent.",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


@app.post("/start-feedback")
async def start_feedback_session(context: FeedbackContext):
    """
    Starts a new feedback session and returns the first question and checkpoint.
    """
    print("---API CALL: /start-feedback---")
    print(context)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {"context": context.dict(), "session_id": session_id ,"first_answer":True}
    
    # Run the agent to get the first question
    response = agent_app.invoke(initial_state, config )
    
    # After the first run, get the state snapshot to find the checkpoint_id
    latest_snapshot = agent_app.get_state(config)
    
    return {
        "session_id": session_id,
        "question": response.get("current_question"),
        # Return the specific checkpoint_id for the next call
        "checkpoint_id": latest_snapshot.config['configurable']['checkpoint_id']
    }

@app.post("/continue-feedback")
async def continue_feedback_session(body: ContinueBody):
    """
    Receives an answer and resumes the conversation from a specific checkpoint.
    """
    print("---API CALL: /continue-feedback---")
    session_id = body.session_id
    user_answer = body.answer
    
    # THIS IS THE CRUCIAL FIX:
    # We tell the agent to load the EXACT "save file" (checkpoint)
    # before resuming. This removes all ambiguity.
    config = {
        "configurable": {
            "thread_id": session_id,
            "checkpoint_id": body.checkpoint_id
        }
    }


    print(agent_app.get_state(config).tasks)
    # Invoke with ONLY the new information the agent is waiting for.


    response = agent_app.invoke(Command(resume={"user_answer": user_answer}),config)
    

    if "summary" in response:
        return {
            "session_id": session_id,
            "question": None,
            "summary": response.get("summary"),
            "output_file": f"feedbacks/feedback_session_{session_id}.csv"
        }
    else:
        # Get the latest state to find the NEW checkpoint_id for the next turn
        latest_snapshot = agent_app.get_state({"configurable": {"thread_id": session_id}})
        return {
            "session_id": session_id,
            "question": response.get("current_question"),
            "checkpoint_id": latest_snapshot.config['configurable']['checkpoint_id']
        }


@app.post("/generate-email", response_class=PlainTextResponse)
async def create_email(request: EmailRequest):
    """
    Receives company details and a prompt, generates an email, and returns it.
    """
    try:
        # Prepare the input for the LangGraph agent
        # The structure must match the `EmailAgentState` TypedDict
        inputs = {
            "company_name":request.company_name,
            "company_type": request.company_type,
            "email_type": request.email_type,
            "prompt": request.prompt,
            "session_id": uuid.uuid4().hex[:8]  # Generate a unique ID for this run
        }
        print(request.company_name)
        # Invoke the agent to run the graph and get the final state
        # The .invoke() method is synchronous, which is fine for this use case.
        # For long-running tasks, you might consider async invocation.
        final_state = email_agent_app.invoke(inputs)

        # Extract the generated email from the final state of the graph
        generated_email = final_state.get("generated_email")

        if not generated_email:
            raise HTTPException(status_code=500, detail="Email generation failed to produce content.")
            
        # Return the generated email as a plain text response
        return PlainTextResponse(content=generated_email)

    except Exception as e:
        # Catch any errors during the agent's execution
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/")
def read_root():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
