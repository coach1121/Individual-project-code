import torch
from torch import nn
from AlexNet import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

folder = 'save_model'
print("Model saving path:", os.path.abspath(os.path.join(folder, 'best_model.pth')))

ROOT_TRAIN = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/Train_Multi-label'
ROOT_VAL = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/Val_Multi-label'


normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

def train_model():
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_VAL, transform=val_transform)
    batch_size = 32  # Increase the batch size if GPU memory allows
    num_workers = 4  # Increase the number of workers for data loading if CPU can handle
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyAlexNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



#
# import torch
# from torch import nn
# from AlexNet import MyAlexNet
# import numpy as np
# from torch.optim import lr_scheduler, RMSprop  # Import RMSprop optimizer
# import os
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import pandas as pd
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# folder = 'save_model'
# print("Model saving path:", os.path.abspath(os.path.join(folder, 'best_model.pth')))
#
# ROOT_TRAIN = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/Train_Multi-label'
# ROOT_VAL = r'D:/Learning Resources/Semester2/MSC Research Project/Code/pythonProject1/Val_Multi-label'
#
# normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor(),
#     normalize])
#
# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     normalize])
#
# def train_model():
#     train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
#     val_dataset = ImageFolder(ROOT_VAL, transform=val_transform)
#     batch_size = 32  # Increase the batch size if GPU memory allows
#     num_workers = 4  # Increase the number of workers for data loading if CPU can handle
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = MyAlexNet().to(device)
#
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = RMSprop(model.parameters(), lr=0.001, momentum=0.9)  # Use RMSprop optimizer instead of SGD

    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        loss, current, n = 0.0, 0.0, 0
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scheduler.step()

        train_loss = loss / n
        train_acc = current / n
        print('train_loss' + str(train_loss))
        print('train_acc' + str(train_acc))
        return train_loss, train_acc

    def val(dataloader, model, loss_fn):
        model.eval()
        loss, current, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                image, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                output = model(image)
                cur_loss = loss_fn(output, y)
                _, pred = torch.max(output, axis=1)
                cur_acc = torch.sum(y == pred) / output.shape[0]
                loss += cur_loss.item()
                current += cur_acc.item()
                n = n + 1

        val_loss = loss / n
        val_acc = current / n
        print('val_loss' + str(val_loss))
        print('val_acc' + str(val_acc))
        return val_loss, val_acc

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoch = 100
    min_acc = 0
    for t in range(epoch):
        print(f"epoch {t+1}\n-----------")
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = val(val_dataloader, model, loss_fn)
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        if val_acc > min_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            min_acc = val_acc
            print(f"Save best model, 第{t+1}轮")
            torch.save(model.state_dict(), os.path.join(folder, 'best_model.pth'))

    patience = 20  # Number of epochs to wait before early stopping
    early_stopping_counter = 0
    best_val_acc = 0.0  # Track the best validation accuracy

    for t in range(epoch):
        print(f"epoch {t + 1}\n-----------")
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = val(val_dataloader, model, loss_fn)
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        if val_acc > best_val_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir(folder)
            best_val_acc = val_acc
            early_stopping_counter = 0  # Reset the counter
            print(f"Save best model, 第{t + 1}轮")
            torch.save(model.state_dict(), os.path.join(folder, 'best_model.pth'))
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Validation performance did not improve for {patience} epochs. Stopping early.")
            break  # Stop training

    torch.save(model.state_dict(), os.path.join(folder, 'last_model.pth'))

    matplot_loss(loss_train, loss_val)
    matplot_acc(acc_train, acc_val)
    print('Done!')

def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Comparison of loss values between training set and verification set")
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("Comparison of accuracy values between training set and verification set")
    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True
    train_model()