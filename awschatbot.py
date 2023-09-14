from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.getenv('LD_LIBRARY_PATH', '')
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['CUDA_RUNTIME_LIB'] = '112'

model_name = "google/flan-t5-xl"

class ChatBot:
    def __init__(self, model_name, init_prompt=[]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=512)
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', load_in_8bit=True, max_memory=max_memory)
        self.conversation_history = init_prompt

    def generate_response(self, input_text):
        self.conversation_history.append(f"User: {input_text}")
        input_with_history = "\n".join(self.conversation_history)
        input_tokens = self.tokenizer.encode(input_with_history, return_tensors='pt').to(self.device)
        generated_tokens = self.model.generate(input_tokens, max_length=500, num_return_sequences=1)
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        self.conversation_history.append(f"Chatbot: {response}")
        
        return response

initial_prompt = [
    "This is a friendly and safe chat session between a user and a computer called Chatbot.",
    "Start of chat:",
    "User: Hello Chatbot, how are you today?",
    "Chatbot: Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to assist you with any information or task you have!",
    "User: What can you do?",
    "Chatbot: I can help answer questions, provide recommendations, help you with calculations, and much more! Feel free to ask me anything.",
    "User: What is the weather like?",
    "Chatbot: I cannot provide real-time information as I am not connected to the internet. However, you can check the weather on a weather website or app by entering your location.",
    "User: Can you tell me a joke?",
    "Chatbot: Sure! Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!",
    "User: That's funny! Can you recommend a good book?",
    "Chatbot: Absolutely, I'd recommend checking out 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari. It offers a compelling overview of the history of our species.",
]


chatbot = ChatBot(model_name, initial_prompt)

st.title("Chatbot")
user_input = st.text_input("You:")
if st.button("Send"):
    response = chatbot.generate_response(user_input)
    st.write(f"Chatbot: {response}")

st.text_area("Chat History:", value="\n".join(chatbot.conversation_history), height=200)
