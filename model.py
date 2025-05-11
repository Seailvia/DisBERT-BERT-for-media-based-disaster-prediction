# model.py
import torch
from transformers import BertForSequenceClassification

class BERTModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(BERTModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
