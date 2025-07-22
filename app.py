from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import sqlite3

from openai import OpenAI
evaluator = OpenAI()


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

DB_PATH = os.getenv("QA_DB_PATH", "qa.db")

def _get_conn():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT UNIQUE,
    answer TEXT
    )
    """)
    return conn

def query_qa(question: str):
    """
    Looks up an exact (case-insensitive) question in the DB.
    Returns {found: bool, answer?: str}.
    """
    conn = _get_conn()
    cur = conn.execute(
        "SELECT answer FROM qa WHERE question = ? COLLATE NOCASE",
        (question,)
    )
    row = cur.fetchone()
    if row:
        return {"found": True, "answer": row[0]}
    else:
        return {"found": False}
    
def upsert_qa(question: str, answer: str):
    """
    Inserts a new Q&A or updates the answer if the question already exists.
    """
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO qa(question, answer)
        VALUES (?, ?)
        ON CONFLICT(question) DO UPDATE SET answer=excluded.answer
        """,
        (question, answer)
    )
    return {"status": "ok"}

    def evaluate_response(question: str, answer: str) -> dict:
        """
        Use a cheaper model (gpt-3.5-turbo) to score & critique an LLM answer.
        Returns: {"score": <1-5>, "feedback": "<one-sentence critique>"}
        """
        prompt = f"""
        You are an evaluator.  Rate the assistant's answer on a 1-5 scale:
        1 = completely incorrect
        5 = fully correct & well-phrased
        Question:
        {question}
        Assistant's answer:
        {answer}
        Respond in JSON:
        {{"score": <int>, "feedback": "<critique>"}}
        """
        resp = evaluator.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

query_qa_json = {
    "name": "query_qa",
    "description": "Look up a stored answer for a user's question. Returns {'found':true, 'answer':...} or {'found':false}.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The full user question to look up"
            }
        },
        "required": ["question"],
        "additionalProperties": False 
    }
}

upsert_qa_json = {
    "name": "upsert_qa",
    "description": "Insert or update a question–answer pair in the Q&A database.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question text to store"
            },
            "answer": {
                "type": "string",
                "description": "The answer text to associate with that question"
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}

# After upsert_qa_json definition, insert:
evaluate_response_json = {
  "name": "evaluate_response",
  "description": "Score and critique an assistant answer (1–5) for correctness & style.",
  "parameters": {
    "type": "object",
    "properties": {
      "question": {"type": "string"},
      "answer":   {"type": "string"}
    },
    "required": ["question","answer"],
    "additionalProperties": False
  }
}


tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json},
        {"type": "function", "function": query_qa_json},
        {"type": "function", "function": upsert_qa_json},
        {"type": "function", "function" : evaluate_response_json},]

class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Rafael Hasanov"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            print(f"Tool result: {result}", flush=True)    # <— add this
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
,Only use record_unknown_question if you absolutely cannot answer even after checking and storing with the Q&A tools.  \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        system_prompt += """
**Before** you do anything else, follow these steps **in order** for *every* user question:

1) Call `query_qa` with JSON exactly:
{"question":"<the user's exact message>"}

   • If it returns {"found":true,"answer":"..."}  
     - THEN reply **only** with that answer and **stop**.  
     - Do **not** call any other tool.

2) Otherwise (it returned {"found":false}):  
   a) Generate the best answer you can.  
   b) Immediately call `upsert_qa` with JSON:
{"question":"<the user's exact message>","answer":"<the answer you just generated>"}  
   c) THEN reply to the user with that answer.  
   d) **Stop**—do **not** call record_unknown_question.

3) **Only if** after steps 1 and 2 you still absolutely have no answer (for instance you generated no content at all), call `record_unknown_question` with:
{"question":"<the user's exact message>"}

**Under no circumstances** call `record_unknown_question` before steps 1 and 2 have fully completed.
"""


        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    