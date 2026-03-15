from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader

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

name = "Mazhar Hussain"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website,\
    particularly questions related to {name}'s career, background, skills and experience. \
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
            You are given a summary of {name}'s background and Curriculum Vitae, which you can use to answer questions. \
                Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
                    If you don't know the answer, say so."

system_prompt+=f"Curriculum Vitae: {content_extract}"
system_prompt+=f"With this context, please chat with the user, always staying in character as {name}"



system_message = [{'role':"system", "content": system_prompt}]
print (system_message)

import gradio as gr

def chatbot(message, history):
    # In newer Gradio, 'history' is already a list of dicts: [{"role": "user", "content": "..."}, ...]
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    response = openai_object.chat.completions.create(
        model="deepseek-chat", 
        messages=messages
    )
    return response.choices[0].message.content

# Remove the type="messages" argument here
gr.ChatInterface(chatbot).launch(share=True)