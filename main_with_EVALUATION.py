from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from pydantic import BaseModel
import json




name = "Mazhar Hussain"
class Evaluation(BaseModel):
    is_acceptable : bool
    feedback : str



reader = PdfReader("me/CV_Clinical_Research.pdf")



pages = reader.pages  # Get the first page
content_extract = f""
for page in pages:
    text = page.extract_text()
    content_extract+=text

#print (content_extract)

#text = page.extract_text()
#prinprint(text)

load_dotenv(override = True)
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_object = OpenAI(api_key = deepseek_key, base_url = "https://api.deepseek.com")



system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website,\
    particularly questions related to {name}'s career, background, skills and experience. \
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
            You are given a summary of {name}'s background and Curriculum Vitae, which you can use to answer questions. \
                Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
                    If you don't know the answer, say so."

system_prompt+=f"Curriculum Vitae: {content_extract}"
system_prompt+=f"With this context, please chat with the user, always staying in character as {name}"

evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:"
evaluator_system_prompt += f"Curriculum Vitae: {content_extract}"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
evaluator_system_prompt += (
    ' Return ONLY valid JSON. '
    'Use this exact format: '
    '{"is_acceptable": true, "feedback": "your feedback here"}'
)

def evaluation_prompt_full(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

def evaluate(reply, message, history) -> Evaluation:
    messages = [{"role":"system","content":evaluator_system_prompt}]+[{"role":"user", "content":evaluation_prompt_full(reply, message, history)}]
    response = openai_object.chat.completions.create(model = "deepseek-chat", messages = messages)
    response = json.loads(response.choices[0].message.content)
    print (response)
    return response

def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai_object.chat.completions.create(model="deepseek-chat", messages=messages)
    return response.choices[0].message.content
system_message = [{'role':"system", "content": system_prompt}]
print (system_message)

import gradio as gr

def chatbot(message, history):
    if "patents" in message:
         system = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    else:
        system = system_prompt
    
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    
    response = openai_object.chat.completions.create(
        model="deepseek-chat", 
        messages=messages
    )
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, history)

    if evaluation["is_acceptable"]:
        gr.Info("Passed evaluation - returning reply")
    else:
        gr.Info("Failed evaluation - retrying")
        gr.Info(evaluation["feedback"])

        reply = rerun(reply, message, history, evaluation["feedback"])

    return reply

# Remove the type="messages" argument here
gr.ChatInterface(chatbot).launch()