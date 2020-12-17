import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import os
from datasets import load_metric
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np
from compute_rouge import calculate_rouge
import nltk

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def shift_tokens_right(input_ids, pad_token_id):
	"""Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
	prev_output_tokens = input_ids.clone()
	index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
	prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
	prev_output_tokens[:, 1:] = input_ids[:, :-1]
	return prev_output_tokens

class BARTSummarization(LightningModule):

    def __init__(self, model_name, tokenizer, learning_rate=1e-4, eval_beams=4):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decoder_start_token_id = None
        self.eval_beams = eval_beams
        self.learning_rate = learning_rate
        self.all_generated = []
        self.all_actual = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        tgt_ids = batch['tgt_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        outputs = self(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss_tensors = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        return loss_tensors


    def training_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        loss_tensors = self._step(batch)
        self.log('train_loss', loss_tensors, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss_tensors}

    def ids_to_clean_text(self, ids):
        gen_text = self.tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def generative_step(self, batch):
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            length_penalty=2.0,
            max_length=256,
            min_length=10,
            no_repeat_ngram_size=3
        )
        generated_text = self.ids_to_clean_text(generated_ids)
        actual_text = self.ids_to_clean_text(batch['tgt_ids'])
        return generated_text, actual_text

    def validation_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        self.log('val_loss', loss_tensors, on_epoch=True, prog_bar=True)
        generated_text, actual_text = self.generative_step(batch)
        self.all_generated += generated_text
        self.all_actual += actual_text
        return {"val_loss": loss_tensors}
    
    def test_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        return {"test_loss": loss_tensors}
    #### Validation end and test end methods


    def validation_epoch_end(self, outputs):
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(calculate_rouge(self.all_generated, self.all_actual,rouge_keys=['rouge1','rouge2', 'rougeL', 'rougeLsum'])) 
        self.all_generated = []
        self.all_actual = []
        self.log("avg_val_loss", avg_val_loss)
        #return {"val_loss": avg_val_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.opt = optimizer
        return [optimizer]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, articles, highlights, tokenizer, length_tokens=None, max_length=512): ####
        self.x = articles
        self.y = highlights
        self.length_tokens = length_tokens
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        x = self.tokenizer.encode_plus(self.x[index].lower(), max_length=self.max_length-1, return_tensors="pt", truncation=True, padding='max_length')
        
        if self.length_tokens==None:
            x_input_ids = x['input_ids'].view(-1)
            x_attention_mask = x['attention_mask'].view(-1)
        else:
            x_input_ids = torch.cat((torch.tensor([self.length_tokens[index]]), x['input_ids'].view(-1)), dim=0)
            x_attention_mask = torch.cat((torch.tensor([1]), x['attention_mask'].view(-1)), dim=0)
        y = self.tokenizer.encode(self.y[index].lower(), max_length=self.max_length, return_tensors="pt", truncation=True, padding='max_length')
        return {'input_ids' : x_input_ids, 'attention_mask' : x_attention_mask, 'tgt_ids' : y.view(-1)}

    def __len__(self):
        return len(self.x)

def read_data(file_path, split, limit=-1):

    articles_path = os.path.join(file_path, split + '_articles.txt')
    highlights_path = os.path.join(file_path, split + '_highlights.txt')
    articles = []
    highlights = []

    with open(articles_path, 'r') as f:
        for l in f:
            articles.append(l.strip())
    with open(highlights_path, 'r') as f:
        for l in f:
            highlights.append(l.strip())
    print(len(articles),len(highlights))
    assert len(articles)==len(highlights)

    if(limit==-1):
        return articles, highlights
    return articles[:limit], highlights[:limit]

def find_length_tokens(summary, bin_l, tokenizer):
    tokens = []
    for s in summary:
        curr_len = len(nltk.word_tokenize(s))
        idx = 0
        while(idx<10):
            if(bin_l[idx]>curr_len):
                break
            idx += 1
        tokens.append(tokenizer.convert_tokens_to_ids(f'<bin_{idx-1}>'))
    return tokens
    



def find_length_control_bins(train_summary):
    all_lengths = [len(nltk.word_tokenize(s)) for s in train_summary]
    N = len(all_lengths)
    bin_l = [0]
    all_lengths.sort()
    for i in range(1,10):
        bin_l.append(all_lengths[int(i*N/10)])
    return bin_l

def add_style_term(articles):
    out = []
    ct = 0
    for article in articles:
        if '(CNN)' in article:
            out.append('<style_cnn> '+article)
            ct += 1
        else:
            out.append('<style_dailymail> '+article)
    print(f'Num CNN:{ct}')
    return out


def main():
    FINETUNE_MODE = 'style'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #DATA_PATH = "/content/drive/My Drive/CNN_DailyMail_Processed"
    #DATA_PATH = "/Users/apple/Downloads/CNN_DailyMail_Processed"
    if FINETUNE_MODE in ['entity', 'length_entity']:
        DATA_PATH = "/home/aakash03/entity_preprocess"
        train_articles, train_hightlights = read_data(DATA_PATH, 'anon_train', -1) ####
        val_articles, val_hightlights = read_data(DATA_PATH, 'anon_test', 20)
        test_articles, test_hightlights = read_data(DATA_PATH, 'anon_test', 20)
    else:
        DATA_PATH = "/home/aakash03/CNN_DailyMail_Processed"

        train_articles, train_hightlights = read_data(DATA_PATH, 'train', -1) ####
        val_articles, val_hightlights = read_data(DATA_PATH, 'val', 20)
        test_articles, test_hightlights = read_data(DATA_PATH, 'test', 20)
    if FINETUNE_MODE in ['style', 'length_style']:
        train_articles = add_style_term(train_articles)
        val_articles = add_style_term(val_articles)
        test_articles = add_style_term(test_articles)

    ####MODEL_NAME = 'sshleifer/distilbart-cnn-6-6' ####
    MODEL_NAME = 'facebook/bart-base'
    BATCH_SIZE = 4
    MAX_EPOCHS = 500
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    checkpoint_dir = None
    if FINETUNE_MODE!='normal':
        checkpoint_dir = 'checkpoint_dir_'+FINETUNE_MODE
        if FINETUNE_MODE in ['length', 'length_entity', 'length_style']:
            length_tokens = [f'<bin_{idx}>' for idx in range(10)]
            tokenizer.add_tokens(length_tokens)
        if FINETUNE_MODE in ['entity', 'length_entity']:
            entity_tokens = [f'@entity{idx}' for idx in range(10)]
            tokenizer.add_tokens(entity_tokens)
        if FINETUNE_MODE in ['style', 'length_style']:
            style_tokens = ['<style_cnn>', '<style_dailymail>']
            tokenizer.add_tokens(style_tokens)

        ####bin_l = find_length_control_bins(train_hightlights)
        if FINETUNE_MODE in ['length', 'length_entity', 'length_style']:
            bin_l = [0, 33, 38, 42, 47, 51, 56, 61, 68, 81]
            train_length_token_list = find_length_tokens(train_hightlights, bin_l, tokenizer)
            val_length_token_list = find_length_tokens(val_hightlights, bin_l, tokenizer)
            test_length_token_list = find_length_tokens(test_hightlights, bin_l, tokenizer)
        else:
            train_length_token_list, val_length_token_list, test_length_token_list = None, None, None


        train_ds = TorchDataset(train_articles, train_hightlights, tokenizer, train_length_token_list)
        val_ds = TorchDataset(val_articles, val_hightlights, tokenizer, val_length_token_list)
        test_ds = TorchDataset(test_articles, test_hightlights, tokenizer, test_length_token_list)
    
    else:
        checkpoint_dir = 'checkpoint_dir_normal'
        train_ds = TorchDataset(train_articles, train_hightlights, tokenizer)
        val_ds = TorchDataset(val_articles, val_hightlights, tokenizer)
        test_ds = TorchDataset(test_articles, test_hightlights, tokenizer)


    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='BARTSummarization')
    summarization_model = BARTSummarization(MODEL_NAME, tokenizer)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=-1,
        verbose=True,
        monitor='avg_val_loss',
        filename='{epoch:02d}-{avg_val_loss:.2f}',
        mode='min'
    )
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=logger, gpus=1, accumulate_grad_batches=4, callbacks=[checkpoint_callback], limit_train_batches=0.1)
    trainer.fit(summarization_model, train_loader, val_loader)

    model = summarization_model.model
    model.eval()
    model.to(device)
    generated_test_summaries = []
    actual_test_summaries = []
    for i, batch in enumerate(test_loader):
        for k in batch:
            batch[k] = batch[k].to(device)
        generated_text, actual_text = summarization_model.generative_step(batch)
        generated_test_summaries += generated_text
        actual_test_summaries += actual_text
    print(calculate_rouge(generated_test_summaries, actual_test_summaries,rouge_keys=['rouge1','rouge2', 'rougeL', 'rougeLsum']))



if __name__ == '__main__':
    main()
