import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# 读取数据
predictions = pd.read_csv('./nlpdata/submission.csv')
ground_truth = pd.read_csv('./nlpdata/test_label.csv')

# 计算 Accuracy
correct_predictions = (predictions['target'] == ground_truth['target']).sum()
total_predictions = len(predictions['target'])
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy * 100:.2f}%")

# 计算 Precision, Recall, F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth['target'], predictions['target'], average=None
)

# 计算 Macro 和 Weighted 平均值
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    ground_truth['target'], predictions['target'], average='macro'
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    ground_truth['target'], predictions['target'], average='weighted'
)

# 打印分类报告
print("\nClassification Report:")
print(classification_report(ground_truth['target'], predictions['target']))

# 打印手动计算的 Precision, Recall, F1
print("\nPer-Class Metrics:")
print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    print(f"{i:<10} {p:.4f}    {r:.4f}    {f:.4f}")

print("\nMacro Avg:")
print(f"Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-Score: {macro_f1:.4f}")

print("\nWeighted Avg:")
print(f"Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-Score: {weighted_f1:.4f}")

# 绘制混淆矩阵
conf_matrix = confusion_matrix(ground_truth['target'], predictions['target'])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()