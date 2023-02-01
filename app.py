import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu

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


# Custom function to calculate character normalised perplexity
def calculate_char_norm_perplexity(model, tokenizer, prompt):
    input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors='pt')).unsqueeze(0)
    outputs = model(input_ids)  
    loss = outputs[0].mean()
    perplexity = torch.exp(loss)
    char_norm_perplexity = perplexity / len(prompt)
    return char_norm_perplexity.item()



st.sidebar.title("Settings")

max_new_tokens = st.sidebar.number_input("Max New Tokens", value=100, min_value=0, max_value=1000)
do_sample= st.sidebar.checkbox("Do Sample?", value=True) # Whether or not to use samplingp 
temperature =st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6) #The value used to module the next token probabilities
top_p= st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0) #  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation


if st.button("Generate"):
    start_time = time.time()
    generator = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)
    generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)[0]["generated_text"]
    end_time = time.time()
    st.success("Generated text: " + generated)
    ## Calculating the time needed to generate the text
    st.write("Time taken: {:.4f} seconds".format(end_time - start_time))

    ##calculates the character normalised perplexity
    char_norm_perplexity = calculate_char_norm_perplexity(model, tokenizer, prompt)
    #st.write("Character Normalised Perplexity: {:.4f}".format(char_norm_perplexity))
    st.markdown(f"<p style='color: green'>Character Normalised Perplexity: {char_norm_perplexity:.4f}</p>", unsafe_allow_html=True)

    with st.expander("What is character normalised perplexity? "):
        st.write("Character Normalized Perplexity is a measure of the quality of a text generation model. It represents how "+
                 "well the model can predict the next character in a sequence given the previous characters. It is a way of "+
                 "evaluating the model's ability to generate coherent and meaningful sequences of characters. The score is "+
                 "calculated by dividing the total number of characters by the total log likelihood of the model's predictions. " +
                 "A lower score indicates a better model, as it means the model is able to predict the next character with high accuracy. ")
        st.write("https://medium.com/towards-data-science/perplexity-in-language-models-87a196019a94 ")
    





