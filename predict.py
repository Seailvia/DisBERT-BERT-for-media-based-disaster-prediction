# predict.py
import torch
import pandas as pd
from model import BERTModel
from data_processing import load_data
from transformers import BertTokenizer

def predict(model, test_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds

def main(train_file, test_file, model_path, submission_file='submission.csv'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    _, test_loader, _, test = load_data(train_file, test_file, tokenizer, max_len=160)
    
    model = BERTModel(num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_preds = predict(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    submission = pd.DataFrame({
        'id': test['id'],
        'target': test_preds
    })
    submission.to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")
