# run_train.py
from train import main
import torch

train_file = './nlpdata/train.csv'
test_file = './nlpdata/test.csv'

trained_model = main(train_file, test_file)

model_path = 'trained_model.pth'
torch.save(trained_model.state_dict(), model_path)
print(f"Model has been saved to {model_path}")