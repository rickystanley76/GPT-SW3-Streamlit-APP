import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

#Setting Hugging face orgAPI key
# .streamlit/secrets.toml should consist org_token= "YOUR_ORGANIZATION_TOKEN"
organization_token = st.secrets['org_token']


# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda" if torch.cuda.is_available() else "cpu"

st.header("Text Generation with GPT-SW3 Model of AI Sweden")

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, auth_token=organization_token)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)

prompt = st.text_input("Enter your prompt:")

st.sidebar.title("Settings")

max_new_tokens = st.sidebar.number_input("Max New Tokens", value=100, min_value=0, max_value=1000)
do_sample= st.sidebar.checkbox("Do Sample?", value=True) # Whether or not to use sampling 
temperature =st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6) #The value used to module the next token probabilities
top_p= st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0) #  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation


if st.button("Generate"):
    generator = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)
    generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)[0]["generated_text"]
    st.success("Generated text: " + generated)
