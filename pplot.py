import matplotlib.pyplot as plt
import seaborn as sns

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 使用seaborn的样式
sns.set_style("whitegrid")
# 增大图形尺寸
plt.figure(figsize=(8, 6)) 

# 训练轮数等数据及绘图代码保持不变
# 训练轮数
epochs = [i for i in range(1, 16)]
# 损失值
losses = [0.5877525638848448, 0.4346571335682092, 0.38173276621598407, 0.35530916753763586, 0.3277079643437657,
          0.30075490300933394, 0.2826704418602081, 0.25989932877520405, 0.2372729818042094, 0.21591196547205838,
          0.20370061207598392, 0.18781285039104592, 0.17120044254147396, 0.1663578376292992, 0.15068446760312817]
# 准确率
accuracies = [0.7046, 0.8188, 0.8444, 0.858, 0.8724, 0.8868, 0.895, 0.9068, 0.9182, 0.9274, 0.9312, 0.9384, 0.946, 0.9488, 0.9536]

# 绘制损失值曲线
plt.plot(epochs, losses, label='Loss', color='red', linestyle='-', marker='o')
# 绘制准确率曲线
plt.plot(epochs, accuracies, label='Accuracy', color='blue', linestyle='--', marker='s')

# 添加标签和标题
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.title('Training Loss and Accuracy', fontsize=14, fontweight='bold')

# 添加图例
plt.legend(fontsize=10, loc='best')

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()