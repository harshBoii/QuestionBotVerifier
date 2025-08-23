# agent.py
import os
import csv
from typing import TypedDict, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


class GraphState(TypedDict):
    context: Dict[str, Any]
    questions: List[str]
    questions_and_answers: List[Dict[str, str]]
    summary: str
    session_id: str
    current_question: str
    user_answer: str
    question_number:int





def generate_questions_node(state: GraphState):
    print("---NODE: GENERATING QUESTIONS---")
    context = state["context"]
    profession = context.get("profession")
    work_experience = context.get("work_experience")
    name=context.get("name")
    
    

    



    prompt = f"""
    Based on the following context about a former employee, generate a list of 5 concise and relevant feedback questions,Using This Que and Ans Try to extract out all DATA FOR SUMMARIZATION.
    The questions should be suitable to ask their ex-boss.

    Context:
    - Profession: {profession}
    - Work Experience: {work_experience}
    - name:{name}

    Return the questions as a numbered list. For example:
    1. First question?
    2. Second question?

    Example of a question:
    How was {name}'s Reputation as {profession} in {work_experience}

    -------VERY IMPORTANT-----------
        The First Question Will be Descriptive But The Remaining 4 Question will be Very Short Word Type That User can answer as "good" , "above average" and so on. 

    """


    response = llm.invoke(prompt)
    questions = [q.strip() for q in response.content.split('\n') if q.strip() and q[0].isdigit()]
    
    print(f"Generated Questions: {questions}")
    state["questions"] = questions
    state["questions"][0]=f"{state["questions"][0]}\n\n PLEASE PROVIDE A DESRIPTIVE SUMMARY OF MORE THAN 50 WORDS"
    state["questions_and_answers"] = []
    return state

def prepare_question_node(state: GraphState):
    """
    Pops the next question from the list and sets it as the current question.
    This node prepares the question that the agent will pause to ask.
    """
    print("---NODE: PREPARING QUESTION---")
    if not state["questions"]:
        state["current_question"] = None
        return state



    question = state["questions"].pop(0)
    state["question_number"]=question[0]
    print(f"Next Question: {question}")
    state["current_question"] = question
    return state

def process_answer_node(state: GraphState):
    """
    This is now a real node. It processes the user's answer from the state
    and adds it to our list of questions and answers.
    """

    print("---NODE: PROCESSING ANSWER---")

    user_answer = interrupt({"question": state["current_question"]})

    if not state["current_question"] or not user_answer:
        return state
        
    last_question = state["current_question"]
    state["questions_and_answers"].append({"question": last_question, "answer": user_answer})
    # Clear the fields for the next turn
    state["current_question"] = None
    state["user_answer"] = user_answer
    print(user_answer)
    return state



def summarize_node(state: GraphState):
    print("---NODE: SUMMARIZING FEEDBACK---")
    q_and_a_formatted = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in state["questions_and_answers"]])

    summary_prompt = f"""
    Based on the following questions and answers from a feedback session with an ex-boss

    --------VERY IMPORTANT---------------
    GENERATE A WELL FORMATTED REPORT of the former employee's profile IN SYNC WITH THE Q AND A.

    THE FOMAT OF THE REPORT WILL BE LIKE :

    Example-

    ** Harsh's Performance Profile**
Harsh consistently demonstrated exceptional performance as a graphic designer, excelling in both technical skills and collaborative work ethic. His contributions significantly impacted project success, consistently exceeding expectations.

**Skills and Expertise**

Design Innovation and Effectiveness: .....

Technical Proficiency: ...

Time Management and Project Delivery: ....

**Work Ethic and Collaboration:**

Collaboration and Communication: ...

Strengths and Weaknesses: ...

Overall Performance: ...




    USE THE REFERENCE FROM QUESTION AND ANSWER AS AN EXAMPLE TO DEMONSTRATE YOUR POINT

    -------------------------------------

    --------------IMPORTANT---------------

    USE The REFERENCE FROM QUE AND ANS like -
    
    example:
    `His expertise in modern frameworks as demonstrated by the successful overhaul of the company's core customer-facing dashboard, resulting in significant improvements in user engagement and page load times when asked about (What specific contributions did Harsh make to projects during their time at the company, and what was the impact of those contributions?) 
    allows him to translate complex requirements into intuitive and visually striking user interfaces as mentioned when asked about Harsh's overall performance as a Senior Frontend Developer .`

    ---------------------------------------
    Focus on skills, work ethic, and overall performance.

    Conversation:
    {q_and_a_formatted}
    """
    summary = llm.invoke(summary_prompt).content
    print(f"Generated Summary: {summary}")
    state["summary"] = summary
    return state

def save_to_csv_node(state: GraphState):
    """
    Saves the questions and answers to a CSV file.
    """
    print("---NODE: SAVING TO CSV---")
    session_id = state["session_id"]
    filename = f"feedbacks/feedback_session_{session_id}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in state["questions_and_answers"]:
            writer.writerow(item)
            
    print(f"Saved conversation to {filename}")
    return state

def ask_again_node(state:GraphState):
    print("------Asking Again Node--------")
    state["current_question"]="Please Enter A response of more than 50 words"
    state["question_number"]="1"
    return state

# 4. Define Conditional Edges
def should_continue(state: GraphState):
    """
    Determines whether to continue the conversation or move to the summary.
    """
    cnt=0
    cnt+=1
    print(f"count is { state["question_number"]}")

    print(len(state["user_answer"]['user_answer'].split(" ")))

    if (state['question_number']=="1" and len(state["user_answer"]['user_answer'].split(" "))<50):
        print("======1st=========")
        state["user_answer"] = None
        return "ask_again"
    if (state['question_number']=="1" and len(state["user_answer"]['user_answer'].split(" "))>50):
        print("======2nd=========")
        print("Setting To false")
        state["user_answer"] = None
        return "prepare_question"    

    if state["questions"] and state['question_number']!="1":
        print("======3rd=========")
        state["user_answer"] = None
        return "prepare_question"    
    else:
        print("Going To Summarize")
        return "summarize"



# 5. Build the Graph
memory = MemorySaver()
workflow = StateGraph(GraphState)

workflow.add_node("generate_questions", generate_questions_node)
workflow.add_node("prepare_question", prepare_question_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("save_to_csv", save_to_csv_node)
workflow.add_node("process_answer", process_answer_node)
workflow.add_node("ask_again",ask_again_node)
workflow.set_entry_point("generate_questions")
workflow.add_edge("generate_questions","prepare_question")
workflow.add_edge("prepare_question", "process_answer")
workflow.add_conditional_edges(
    "process_answer",
    should_continue,
    {
        "prepare_question": "prepare_question",
        "summarize": "summarize",
        "ask_again":"ask_again"
    },
)
workflow.add_edge("ask_again","process_answer")
workflow.add_edge("summarize", "save_to_csv")
workflow.add_edge("save_to_csv", END)


agent_app = workflow.compile(checkpointer=memory)



