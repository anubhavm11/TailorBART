import streamlit as st
import numpy as np
import torch
from transformers import BartTokenizer
from model.finetune_bart_model import BARTSummarization
import re
import nltk
import spacy
from collections import Counter

MODEL_NAME = 'facebook/bart-base'
num_length_bins = 10
num_entities = 10
max_length = 512
CHECKPOINT_PATH_0 = 'ckpt/base_model_latest.ckpt'
CHECKPOINT_PATH_1 = 'ckpt/length.ckpt'
CHECKPOINT_PATH_2 = 'ckpt/entity.ckpt'
CHECKPOINT_PATH_3 = 'ckpt/length_entity.ckpt'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not hasattr(st, 'out'):
	st.out = ""

if not hasattr(st, 'out_len'):
	st.out_len = ""

if not hasattr(st, 'prev_state'):
	st.prev_state = ""

if not hasattr(st, 'cur_text'):
	st.cur_text = ""

if not hasattr(st, 'entity_dict'):
	st.entity_dict = {}

if not hasattr(st, 'entity'):
	st.entity = ''

if not hasattr(st, 'tokenizer0'):
	st.tokenizer0 = BartTokenizer.from_pretrained(MODEL_NAME)

if not hasattr(st, 'tokenizer1'):
	st.tokenizer1 = BartTokenizer.from_pretrained(MODEL_NAME)
	length_tokens = [f'<bin_{idx}>' for idx in range(num_length_bins)]
	st.tokenizer1.add_tokens(length_tokens)

if not hasattr(st, 'tokenizer2'):
	st.tokenizer2 = BartTokenizer.from_pretrained(MODEL_NAME)
	entity_tokens = [f'@entity{idx}' for idx in range(num_entities)]
	st.tokenizer2.add_tokens(entity_tokens)

if not hasattr(st, 'tokenizer3'):
	st.tokenizer3 = BartTokenizer.from_pretrained(MODEL_NAME)
	length_tokens = [f'<bin_{idx}>' for idx in range(num_length_bins)]
	st.tokenizer3.add_tokens(length_tokens)
	entity_tokens = [f'@entity{idx}' for idx in range(num_entities)]
	st.tokenizer3.add_tokens(entity_tokens)

if not hasattr(st, 'model0'):
	st.model0 = BARTSummarization.load_from_checkpoint(CHECKPOINT_PATH_0,model_name=MODEL_NAME, tokenizer=st.tokenizer0)
	st.model0.to(DEVICE)

if not hasattr(st, 'model1'):
	st.model1 = BARTSummarization.load_from_checkpoint(CHECKPOINT_PATH_1,model_name=MODEL_NAME, tokenizer=st.tokenizer1)
	st.model1.to(DEVICE)

if not hasattr(st, 'model2'):
	st.model2 = BARTSummarization.load_from_checkpoint(CHECKPOINT_PATH_2,model_name=MODEL_NAME, tokenizer=st.tokenizer2)
	st.model2.to(DEVICE)

if not hasattr(st, 'model3'):
	st.model3 = BARTSummarization.load_from_checkpoint(CHECKPOINT_PATH_3,model_name=MODEL_NAME, tokenizer=st.tokenizer3)
	st.model3.to(DEVICE)

if not hasattr(st, 'nlp'):
	st.nlp = spacy.load("en_core_web_sm")

def replace(old_phrase, new_phrase, sentence):
	old_phrase_cp = re.escape(old_phrase)
	old_phrase_cp = "((?<=[^a-zA-Z0-9])|(?<=^))" + old_phrase_cp + "((?=[^a-zA-Z0-9])|(?=$))"
	return re.sub(old_phrase_cp, new_phrase, sentence, flags=re.IGNORECASE)

def encode_text_len(input_text, length_bin):
	x = st.model1.tokenizer.encode_plus(input_text.lower(), max_length=512-1, return_tensors="pt", truncation=True, padding='max_length')
	x_input_ids = torch.cat((torch.tensor([st.model1.tokenizer.convert_tokens_to_ids(length_bin)]), x['input_ids'].view(-1)), dim=0)
	x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)
	return {"input_ids": x_input_ids.view(1, -1), "attention_mask": x_attention_mask.view(1, -1)}

def encode_text_ent(input_text):
	input_text_cp = str(input_text.lower())
	for key, value in st.entity_dict.items():
		input_text_cp = replace(key, value, input_text_cp)

	input_text_cp = ' '.join([st.entity_dict[item] for item in st.entity]) + ' ' + input_text_cp

	x = st.model2.tokenizer.encode_plus(input_text_cp, max_length=512-1, return_tensors="pt", truncation=True, padding='max_length')
	#x_input_ids = torch.cat((torch.tensor([st.model2.tokenizer.convert_tokens_to_ids(st.entity_dict[st.entity])]), x['input_ids'].view(-1)), dim=0)
	#x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)
	#print(x_input_ids)
	#print(st.entity)
	return {"input_ids": x['input_ids'].view(1, -1), "attention_mask": x['attention_mask'].view(1, -1)}

def encode_text_both(input_text, length_bin):
	input_text_cp = str(input_text.lower())
	for key, value in st.entity_dict.items():
		input_text_cp = replace(key, value, input_text_cp)
	input_text_cp = ' '.join([st.entity_dict[item] for item in st.entity])  + ' ' + input_text_cp

	x = st.model3.tokenizer.encode_plus(input_text_cp, max_length=512-1, return_tensors="pt", truncation=True, padding='max_length')
	x_input_ids = torch.cat((torch.tensor([st.model3.tokenizer.convert_tokens_to_ids(length_bin)]), x['input_ids'].view(-1)), dim=0)
	x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)
	return {"input_ids": x_input_ids.view(1, -1), "attention_mask": x_attention_mask.view(1, -1)}

def compute_output(x, model, eval_beams = 4):
	for key in x:
			x[key] = x[key].to(DEVICE)
	generated_ids = model.model.generate(
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
	output = model.ids_to_clean_text(generated_ids)[0]
	return output

def compute_summary(input_text, length_bin = None , entity_token = None, eval_beams=4):
	output = None
	if (not length_bin) and (not entity_token):
		x = st.model0.tokenizer.encode_plus(input_text.lower(), max_length=512, return_tensors="pt", truncation=True, padding='max_length')
		output = compute_output(x, st.model0, eval_beams)

	elif length_bin and (not entity_token):
		x = encode_text_len(input_text, f'<bin_{length_bin-1}>')
		output = compute_output(x, st.model1, eval_beams)
		
	elif (not length_bin) and (entity_token):
		x = encode_text_ent(input_text)
		output = compute_output(x, st.model2, eval_beams)
		#print(output)
		for key, value in st.entity_dict.items():
			output = output.replace(value, '**'+key+'**')
		#print(output)
	
	else:
		x = encode_text_both(input_text, f'<bin_{length_bin-1}>')
		output = compute_output(x, st.model3, eval_beams)
		#print(output)
		for key, value in st.entity_dict.items():
			output = output.replace(value, '**'+key+'**')

	output = output.replace("<s>","")
	output = output.replace("</s>","")
	return output, len(nltk.word_tokenize(output))
	
def get_entity_list(article):
	allowed_entitylabels = ["PERSON", "GPE", "ORG", "WORK_OF_ART", "GEO", "NORP", "EVENT"]
	articles_entity = st.nlp(article)
	entities = [X.text for X in articles_entity.ents if X.label_ in allowed_entitylabels]
	entities_freq = Counter(entities).most_common(num_entities)
	entities_limited = []
	for entity_tuple in entities_freq:
		entities_limited.append(entity_tuple[0].lower())

	final_entities = set()
	anon_dict = dict()
	index = 0
	for entity in entities_limited:
		final_entities.add(entity)
		anon_dict[entity] = f'@entity{index}'
		index +=1
	return anon_dict

st.title('Controlled summarization using TailorBART')

cur_state = st.radio("Choose control", ('No control','Length control', 'Entity control', 'Length and Entity control'))
st.markdown('Enter a **String input** and select **Control** to see what **output summary** the TailorBART summarizer gives.')

sample_text = "Ever noticed how plane seats appear to be getting smaller and smaller? With increasing numbers of people taking to the skies, some experts are questioning if having such packed out planes is putting passengers at risk. They say that the shrinking space on aeroplanes is not only uncomfortable - it's putting our health and safety in danger. More than squabbling over the arm rest, shrinking space on planes putting our health and safety in danger? This week, a U.S consumer advisory group set up by the Department of Transportation said at a public hearing that while the government is happy to set standards for animals flying on planes, it doesn't stipulate a minimum amount of space for humans. 'In a world where animals have more rights to space and food than humans,' said Charlie Leocha, consumer representative on the committee. 'It is time that the DOT and FAA take a stand for humane treatment of passengers.' But could crowding on planes lead to more serious issues than fighting for space in the overhead lockers, crashing elbows and seat back kicking? Tests conducted by the FAA use planes with a 31 inch pitch, a standard which on some airlines has decreased . Many economy seats on United Airlines have 30 inches of room, while some airlines offer as little as 28 inches . Cynthia Corbertt, a human factors researcher with the Federal Aviation Administration, that it conducts tests on how quickly passengers can leave a plane. But these tests are conducted using planes with 31 inches between each row of seats, a standard which on some airlines has decreased, reported the Detroit News. The distance between two seats from one point on a seat to the same point on the seat behind it is known as the pitch. While most airlines stick to a pitch of 31 inches or above, some fall below this. While United Airlines has 30 inches of space, Gulf Air economy seats have between 29 and 32 inches, Air Asia offers 29 inches and Spirit Airlines offers just 28 inches. British Airways has a seat pitch of 31 inches, while easyJet has 29 inches, Thomson's short haul seat pitch is 28 inches, and Virgin Atlantic's is 30-31."

input_text = st.text_area('Raw String input', sample_text, height=25)

if cur_state != st.prev_state:
	st.prev_state = cur_state
	st.entity_dict = {}
	st.entity = ''
	st.out = ''
	st.out_len = ''

if cur_state == 'No control':

	if st.button('Submit'):
		st.out, out_len = compute_summary(input_text)
		st.out_len = "*Number of words* = "+str(out_len)

elif cur_state == 'Length control':
	length_bin = st.slider('Length', 1, 10, 1)

	if st.button('Submit'):
		st.out, out_len = compute_summary(input_text, length_bin = length_bin)
		st.out_len = "*Number of words* = "+str(out_len)

elif cur_state == 'Entity control':
	# length_bin = st.slider('Length', 1, 10, 1)

	if st.button('Generate Entities'):
		st.cur_text = input_text
		st.entity_dict = get_entity_list(input_text)
		st.out = ''
		st.out_len = ''

	if st.entity_dict:
		st.entity = st.multiselect("Choose entity:", [key for key, value in st.entity_dict.items()])


	if st.entity_dict and st.button('Submit'):
		if st.cur_text == input_text:
			st.out, out_len = compute_summary(input_text, entity_token = [st.entity_dict[entity] for entity in st.entity])
			#st.out = st.out.replace(st.entity, "**"+st.entity+"**")
			st.out_len = "*Number of words* = "+str(out_len)
		else:
			st.markdown("Your input text has changed. Please generate new entities.")

elif cur_state == 'Length and Entity control':
	length_bin = st.slider('Length', 1, 10, 1)

	if st.button('Generate Entities'):
		st.cur_text = input_text
		st.entity_dict = get_entity_list(input_text)
		st.out = ''
		st.out_len = ''

	if st.entity_dict:
		st.entity = st.multiselect("Choose entity:", [key for key, value in st.entity_dict.items()])

	if st.entity_dict and st.button('Submit'):
		if st.cur_text == input_text:
			st.out, out_len = compute_summary(input_text, length_bin = length_bin, entity_token = [st.entity_dict[entity] for entity in st.entity])
			#st.out = st.out.replace(st.entity, "**"+st.entity+"**")
			st.out_len = "*Number of words* = "+str(out_len)
		else:
			st.markdown("Your input text has changed. Please generate new entities.")

st.write("")
st.markdown("**Output Summary**")
st.markdown(st.out)
st.markdown(st.out_len)

