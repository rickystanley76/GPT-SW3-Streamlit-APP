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

st.sidebar.title("Settings")

max_new_tokens = st.sidebar.number_input("Max New Tokens", value=100, min_value=0, max_value=1000)
do_sample= st.sidebar.checkbox("Do Sample?", value=True) # Whether or not to use sampling 
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

    ##BLEU Score Calculation
    prompt = prompt.split()
    generated = generated.split()
    bleu_score = sentence_bleu([prompt], generated)
    st.write("BLEU Score(1 is perfect match):", bleu_score)

    with st.expander("What is BLEU score? "):
        st.write("BLEU (Bilingual Evaluation Understudy) is a widely used automatic evaluation metric in Natural Language Processing (NLP) "+
                 "for evaluating the quality of machine-generated text in comparison to reference human-generated text. It was developed  "+
                 "by Kishore Papineni et al. in 2002. ")
        st.write("BLEU is a comparison-based evaluation metric that calculates the n-gram precision between a machine-generated text and  "+
                 "a reference text.")
        st.write("The interpretation of BLEU score is based on the number of matching n-grams between the generated text and reference text. "+
                 "The score ranges from 0 to 1, with 1 indicating a perfect match. A higher score indicates that the generated text is closer  "+
                 "to the reference text. The exact interpretation of the score varies based on the order of n-grams (i.e. 1-gram, 2-gram, "+
                 "3-gram, etc.) used in the calculation. ")  
        st.write("For example, a BLEU score of 0.4 means that 40% of the n-grams in the generated text match the n-grams in the reference "+
                 "text. A score of 0.6 means 60% match, and so on. The exact value of the score will depend on the specifics of the task,  "+
                 "the quality of the reference text, and the performance of the text generation model.")   
        st.write("https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1 ")     

    





