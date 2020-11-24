import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import os
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np
from compute_rouge import calculate_rouge
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import spacy
import re
from tqdm import tqdm
import en_core_web_lg
nlp = en_core_web_lg.load()

def replace(old_phrase, new_phrase, sentence):
	old_phrase = re.escape(old_phrase)
	old_phrase = "((?<=[^a-zA-Z0-9])|(?<=^))" + old_phrase + "((?=[^a-zA-Z0-9])|(?=$))"
	return re.sub(old_phrase, new_phrase, sentence, flags=re.IGNORECASE)

def anonymize_data(articles, highlights):
	anon_articles = []
	anon_highlights = []
	anon_entities = []

    allowed_entitylabels = ["PERSON", "GPE", "ORG", "WORK_OF_ART", "GEO", "NORP", "EVENT"]
	print("length of the articles is ", len(highlights))
	for article, highlight in tqdm(zip(articles, highlights)):
		articles_entity = nlp(article)
		highlight_entity = nlp(highlight)
		entities = [X.text for X in articles_entity.ents if X.label_ in allowed_entitylabels]
		entities.extend([X.text for X in highlight_entity.ents if X.label_ in allowed_entitylabels])
		entities_freq = Counter(entities).most_common(10)
		entities_limited = []
		for entity_tuple in entities_freq:
			entities_limited.append(entity_tuple[0])

		final_entities = set()
		anon_dict = dict()
		index = 0
		for entity in entities_limited:
			final_entities.add(entity)
			anon_dict[entity] = f'@entity{index}'
			index +=1

		for key, value in anon_dict.items():
			article = replace(key, value, article)
			highlight = replace(key, value, highlight)

		entity_line = " ".join(set(re.findall(r"@entity[0-9]+", highlight)))
		anon_entities.append(entity_line)
		anon_articles.append(article)
		anon_highlights.append(highlight)
