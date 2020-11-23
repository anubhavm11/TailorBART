import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
import os
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np

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
        #### Check this
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