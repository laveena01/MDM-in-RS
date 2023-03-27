import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
def imshow(img, title=None):
    img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)



data_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


test_data_transform = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=r'D:\CODE_AND_RESULTS\dataset\data_airplane\train',
                                     transform=data_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=16, shuffle=True,
                                             num_workers=2)


dataset_sizes = len(train_dataset)
classes = ('background', 'aiplane')
