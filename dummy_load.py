from ollama import chat
from ollama import ChatResponse
import json

# response: ChatResponse = chat(model='XianYu_bi/DeepSeek-R1-Distill-Qwen-14B-Q3_K_M:latest', messages=[
def call():
        response = chat(model='qwen3:8b', messages=[
        {
        'role': 'user',
        'content': "Buatkan saya puisi romantis yang singkat namun dengan nuansa yang tragis dan menyentuh hati.",
        },
        ],
        stream=True)

        # print(response.eval_count) 
        # print(response['prompt_eval_count']) 
        # print(response['message']['content'])
        for chunk in response:
                print(chunk['message']['content'], end='', flush=True)

while True:
    call()