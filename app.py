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
max_new_tokens = st.number_input("Max new tokens:", value=100, min_value=0, max_value=1000)
do_sample = st.checkbox("Do sample?", value=True)
temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.6)
top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=1.0)

if st.button("Generate"):
    generator = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)
    generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)[0]["generated_text"]
    st.success("Generated text: " + generated)
