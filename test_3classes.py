import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import seaborn as sns  # for visualization

# Import your model class
from AlexNet import MyAlexNet

# Mapping relationships
mapping = {
    1: "HW",
    2: "HW,FW",
    3: "HW,ABW",
    4: "HW,DC",
    5: "HW,HFCK",
    6: "HW,FW,ABW",
    7: "HW,FW,MK",
    8: "FW",
    9: "FW,ABW",
    10: "FW,MK",
    11: "ABW",
    12: "ABW,MK",
    13: "SMC",
    14: "MK",
    15: "DC",
    16: "HFCK",
    17: "SPW",
    18: "Noise",
    19: "HW,MK"
}

# Define data preprocessing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Set the path to the test data folder
ROOT_TEST = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/Test_Multi-label'

# Load the test dataset
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Increased batch_size

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyAlexNet().to(device)
model.load_state_dict(
    torch.load("C:/Users/dell/Desktop/Multi-label_results/AlexNet/0.0001/best_model-p.pth"))

# Use the model to predict the validation data
model.eval()
all_probs = []
all_labels = []
all_confidences = []  # Store confidence scores

with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Calculate confidence scores
        confidences = torch.max(probabilities, dim=1)[0]

        all_probs.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

# Map back to original labels
mapped_labels = [mapping[label + 1] for label in all_labels]

# Filter out labels involving HW or FW
hw_fw_indices = [i for i, label in enumerate(mapped_labels) if "HW" in label or "FW" in label]
hw_fw_mapped_labels = [mapped_labels[i] for i in hw_fw_indices]
hw_fw_preds = [np.argmax(all_probs, axis=1)[i] for i in hw_fw_indices]
hw_fw_confidences = [all_confidences[i] for i in hw_fw_indices]

# Map predicted labels from numbers back to strings
hw_fw_preds_strings = [mapping[pred + 1] for pred in hw_fw_preds]

# Calculate metrics for labels involving HW and FW
accuracy_hw_fw = accuracy_score(hw_fw_mapped_labels, hw_fw_preds_strings)
precision_hw_fw = precision_score(hw_fw_mapped_labels, hw_fw_preds_strings, average='weighted', zero_division=1)
recall_hw_fw = recall_score(hw_fw_mapped_labels, hw_fw_preds_strings, average='weighted', zero_division=1)
f1_hw_fw = f1_score(hw_fw_mapped_labels, hw_fw_preds_strings, average='weighted', zero_division=1)

# Print results with confidence scores
for i in range(len(hw_fw_mapped_labels)):
    print(
        f"Label: {hw_fw_mapped_labels[i]}, Predicted: {hw_fw_preds_strings[i]}, Confidence: {hw_fw_confidences[i]:.2f}")

print("\nMetrics for labels involving HW and FW:")
print(f"Accuracy: {accuracy_hw_fw:.2f}")
print(f"Precision: {precision_hw_fw:.2f}")
print(f"Recall: {recall_hw_fw:.2f}")
print(f"F1 Score: {f1_hw_fw:.2f}")

# Plot confusion matrix for labels involving HW and FW
conf_matrix_hw_fw = confusion_matrix(hw_fw_mapped_labels, hw_fw_preds_strings, labels=list(mapping.values()))
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_hw_fw, annot=True, fmt="d", cmap="Blues", xticklabels=list(mapping.values()),
            yticklabels=list(mapping.values()))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for labels involving HW and FW')
plt.show()
