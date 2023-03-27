
import torch
import numpy as np
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes,
                model_path, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for batch_idx, (_, inputs, targets) in enumerate(dataloaders):
            for inputs, targets in dataloaders:
                # print(len(inputs))
                inputs = inputs.to(device)
                targets = targets.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = model.get_loss(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            # if phase == 'test' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            # best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.eval()
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, model_path)
    model.load_state_dict(best_model_wts)
    return model

def get_optim(model):
    lr = 0.001
    added_layers = ['fc', 'cls', 'classifier']
    weight_list = []
    bias_list = []
    added_weight_list = []
    added_bias_list = []
    print('\n following parameters will be assigned 10x learning rate:')
    for name, value in model.named_parameters():
        if any([x in name for x in added_layers]):
            print(name)
            if 'weight' in name:
                added_weight_list.append(value)
            elif 'bias' in name:
                added_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)

            elif 'bias' in name:
                bias_list.append(value)
            else:
                weight_list.append(value)

    optimizer = optim.SGD([{'params': weight_list, 'lr': lr},
                           {'params': bias_list, 'lr': lr * 2},
                           {'params': added_weight_list, 'lr': lr * 10},
                           {'params': added_bias_list, 'lr': lr * 20}],
                          momentum=0.9, weight_decay=0.0005, nesterov=True)
    return optimizer


if __name__ == "__main__":
    num_maps = [4, 8, 16, 32]
    cos_alpha = [0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2]
    name = 'airplane'
    # load data
    from new_methods.utils.load_data import get_dataloader
    dataloaders, dataset_sizes, categories = get_dataloader(name=name, batch_size=16)
    for select_model in ['DA_PAM', 'DA']:
        for cos in cos_alpha:
            for k in num_maps:
                if name == 'VHR10':
                    path = '../utils/model_trained/params_multi10_'
                elif name == 'DIOR':
                    path = '../utils/model_trained/params_multi20_'
                else:
                    path = '../utils/model_trained/params_single2_'
                # load model
                if select_model == 'CAM':
                    from new_methods.model.resnet import model

                    model_ft = model(pretrained=True, num_classes=len(categories))

                elif select_model == 'DA':
                    from new_methods.model.resnet_DA import model

                    model_ft = model(pretrained=True, num_classes=len(categories), cos_alpha=cos, num_maps=k)
                else:
                    from new_methods.model.resnet_DA_PAM import model

                    model_ft = model(pretrained=True, num_classes=len(categories), cos_alpha=cos, num_maps=k)
                model_ft = model_ft.to(device)


                # load train super parameters'
                optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
                # optimizer_ft = get_optim(model_ft)
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

                # train model

                model_path = path + select_model + '_'+str(cos)+'_'+ str(k) +'.pkl'
                train_model(model_ft, optimizer_ft, exp_lr_scheduler, dataloaders,
                        dataset_sizes, model_path, num_epochs=70)