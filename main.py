# main.py

import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langgraph.types import Command
# Import the agent_app from your corrected agent.py
from agent import agent_app
from fastapi.middleware.cors import CORSMiddleware



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

@app.get("/")
def read_root():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
