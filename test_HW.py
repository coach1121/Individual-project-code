import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scipy.stats import sem, t

# 导入你的模型类
# from VGG16 import vgg16
from AlexNet import MyAlexNet

# 定义类别标签
classes = [
    "HW",
    "NoHW",
]

# 定义数据预处理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 设置测试数据文件夹路径
ROOT_TEST = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/TEST_2classes_HW_trans'

# 加载测试数据集
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyAlexNet().to(device)
model.load_state_dict(
    torch.load("D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/save_model/best_model.pth"))

# 获取预测概率和真实标签
all_probs = []
all_labels = []


# Calculate the confidence interval for accuracy
def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m, h

model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        y = y.item()

        # 获取预测概率
        pred = model(x)
        probs = torch.softmax(pred, dim=1)[0].cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)


# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
precision = precision_score(all_labels, np.argmax(all_probs, axis=1))
recall = recall_score(all_labels, np.argmax(all_probs, axis=1))
f1 = f1_score(all_labels, np.argmax(all_probs, axis=1))

# Calculate confidence intervals
accuracy_mean, accuracy_error = confidence_interval(all_labels == np.argmax(all_probs, axis=1))

# Print metrics with confidence intervals
print(f"Accuracy: {accuracy_mean:.2f} ± {accuracy_error:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 绘制ROC曲线
plt.figure(figsize=(10, 7))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
