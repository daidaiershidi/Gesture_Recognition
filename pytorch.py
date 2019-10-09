# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import argparse
import torch.backends.cudnn as cudnn

num_classes = 3

def train_model(model, best_acc,start_epoch,criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() 
    for epoch in range(start_epoch,num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train','test']:#, 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
			
			# Save checkpoint.
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if phase == 'train':
                train_acc = epoch_acc
            if phase == 'test' and epoch_acc > best_acc and train_acc > 0.9:
                best_acc = epoch_acc
                print('more good..')
                torch.save(model.module.state_dict(), './checkpoint/ckpt.t7')
            
        end = time.time()
        print('using time is {:.0f}m {:.0f}s'.format((end-start) // 60, (end-start) % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    return model
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='PyTorch car images Training')
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint')
    parser.add_argument('--test','-t', default=False, action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop(180),
            transforms.ColorJitter(1,1,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(x, data_transforms[x]) for x in ['train','test']}#,'test']}#, 'val']}train/1,  /2
    classes = [d for d in os.listdir('train') if os.path.isdir(os.path.join('train', d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class ={ i:classes[i] for i in range(len(classes))}
    print("class to id:{}".format(class_to_idx))
    print("id to class:{}".format(idx_to_class))
    print(image_datasets['test'])
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=2,
                                                    shuffle=True,
                                                    num_workers=2) for x in ['train', 'test']}#, 'test']}#, 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print('Test dataset size : {}'.format(dataset_sizes['test']))
    print('Train dataset size : {}'.format(dataset_sizes['train']))

    if args.resume:
        print('==> Resuming from checkpoint..')
        acc = 0
        startepoch = 0
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        model_ft = models.resnet18(pretrained=True)
        num_ftrs1 = model_ft.fc.in_features#512
        model_ft.fc = nn.Linear(num_ftrs1, num_classes)
        model_ft.load_state_dict(torch.load('./checkpoint/ckpt.t7'))
        startepoch = 0#checkpoint['epoch']
    else:
        print('==> Building model..')
        acc = 0
        startepoch = 0
        model_ft = models.resnet18(pretrained=True)
        num_ftrs1 = model_ft.fc.in_features#512
        model_ft.fc = nn.Linear(num_ftrs1, num_classes)

    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    if use_gpu:
            print('can use GPU!!!')
            model_ft = model_ft.cuda()
            model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    if args.test:
        data_transforms = {
            'pred': transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(70),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets['pred']=datasets.ImageFolder('pred', data_transforms['pred']) 
        print(image_datasets['pred'])
        dataloders['pred']=torch.utils.data.DataLoader(image_datasets['pred'],
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=1) 
        data_gen = enumerate(dataloders['pred'])
        output=[]
        import time
        import numpy as np
        proc_start_time = time.time()
        for i, data in data_gen:
            input_var = torch.autograd.Variable(data[0])
            # compute output
            rst= model(input_var)
            rst=rst.detach().cpu().numpy().copy()
            # measure accuracy and record loss
            end = time.time()
            output.append(rst.reshape(1,num_class))
            cnt_time = time.time() - proc_start_time
        image_pred = [np.argmax(x[0]) for x in output]
        print('results is ',idx_to_class[image_pred],' ')
    else:
        model_ft = train_model(model=model_ft,best_acc=acc,start_epoch=startepoch,
                            criterion=criterion,
                            optimizer=optimizer_ft,
                            scheduler=exp_lr_scheduler,
                            num_epochs=30)
                            