# run_predict.py
from predict import main

train_file = './nlpdata/train.csv'
test_file = './nlpdata/test.csv'
model_path = './8017BERT/trained_model.pth'  # 替换为实际的模型文件路径
submission_file = 'submission.csv'

main(train_file, test_file, model_path, submission_file)