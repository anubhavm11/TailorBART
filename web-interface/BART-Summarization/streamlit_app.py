import streamlit as st
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import re

MODEL_NAME = 'sshleifer/distilbart-cnn-6-6'
num_length_bins = 10
max_length = 512

if not hasattr(st, 'out'):
	st.out = ""

if not hasattr(st, 'out_len'):
	st.out_len = ""

if not hasattr(st, 'model'):
	st.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

if not hasattr(st, 'tokenizer'):
	st.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
	entity_tokens = [f'<bin_{idx}>' for idx in range(num_length_bins)]
	st.tokenizer.add_tokens(entity_tokens)

# @st.cache
# def get_model_tokenizer():
# 	model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
# 	tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
# 	entity_tokens = [f'<bin_{idx}>' for idx in range(num_length_bins)]
# 	tokenizer.add_tokens(entity_tokens)
# 	return model, tokenizer

def compute_summary(input_text, length_bin, eval_beams=4):
	# model, tokenizer = get_model_tokenizer()

	x = st.tokenizer.encode_plus(input_text.lower(), max_length=max_length, return_tensors="pt", truncation=True, padding='max_length')
	generated_ids = st.model.generate(
            x["input_ids"],
            attention_mask=x["attention_mask"],
            use_cache=True,
            decoder_start_token_id=None,
            num_beams=eval_beams,
            length_penalty=2.0,
            max_length=256,
            min_length=10,
            no_repeat_ngram_size=3
        )
	output = st.tokenizer.batch_decode(generated_ids)[0]
	output = output.replace("<s>","")
	output = output.replace("</s>","")

	return output
	
def compute_summary_length(input_text, length_bin, eval_beams=4):
	# model, tokenizer = get_model_tokenizer()

	x = st.tokenizer.encode_plus(input_text.lower(), max_length=max_length-1, return_tensors="pt", truncation=True, padding='max_length')
	x_input_ids = torch.cat((torch.tensor([st.tokenizer.convert_tokens_to_ids(f'<bin_{length_bin-1}>')]), x['input_ids'].view(-1)), dim=0)
	x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)
	generated_ids = st.model.generate(
            x_input_ids,
            attention_mask=x_attention_mask,
            use_cache=True,
            decoder_start_token_id=None,
            num_beams=eval_beams,
            length_penalty=2.0,
            max_length=256,
            min_length=10,
            no_repeat_ngram_size=3
        )
	output = st.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
	output = output.replace("<s>","")
	output = output.replace("</s>","")

	return output

st.title('Controlled summarization using BART')
st.markdown('Enter a **String input** and select **Length** to see what **output summary** the BART summarizer gives.')

input_text = st.text_area('Raw String input', 'Life of Brian', height=25)
length_bin = st.slider('Length', 1, 10, 1)
if st.button('Submit'):
	st.out = compute_summary(input_text,length_bin)
	# st.out = compute_summary_length(input_text,length_bin)

	st.out_len = "*Number of words* = "+str(len(re.findall(r'\w+',st.out)))

st.write("")
st.markdown("**Output Summary**")
st.write(st.out)
st.markdown(st.out_len)

