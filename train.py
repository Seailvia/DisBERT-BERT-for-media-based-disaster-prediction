# train.py
import torch
from transformers import AdamW
import torch.nn as nn
from model import BERTModel
from data_processing import load_data
from transformers import BertTokenizer

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader.dataset)
    
    return avg_loss, accuracy

def main(train_file, test_file, epochs=15, max_len=160, lr=2e-6):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, test_loader, train, test = load_data(train_file, test_file, tokenizer, max_len)
    
    model = BERTModel(num_labels=2)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        loss, accuracy = train_model(model, train_loader, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss}, Accuracy: {accuracy}')
        
    return model
