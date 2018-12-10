import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch 
from skimage.transform import resize
from scipy.misc import imresize 

from tensorboardX import SummaryWriter

import os 
import sys 
import time 
import argparse
import datetime
import copy 

from spectralnet import SpectralNet
from GeoTiffDataset import DatasetFolder


# Return network and filename 
def getNetwork(num_bands):
    net = SpectralNet(num_bands=num_bands)
    tm = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = 'SpectralNet-'+str(num_bands)+tm 
    return net, file_name


# Training 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() 
    input_num = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)

        # Each epoch has a training and validation phase 
        for phase in ['train']: #, 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train() # Set model to training mode 
            else:
                model.eval()  # Set model to evalution mode 

            running_loss = 0.0 
            running_corrects = 0.0 

            # iterate over data 

            for inputs, labels, _ in data_loaders[phase]:
                #print(inputs.size())
                inputs = inputs.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.to(device)

                # zero the parameter gradients 
                optimizer.zero_grad()

                # forward pass, track history only in train mode 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backprop only if training phase 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() 
                        writer.add_scalar('data/train_loss', loss.item(), input_num)
                        input_num += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == "val":
                writer.add_scalar("data/acc", epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model 
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc 
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print() 
    
    time_elapsed = time.time() - since 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation Acc: {:.4f}".format(best_acc))

    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model 

def test_model(model):
    running_corrects = 0

    model.eval()
    for inputs, labels, _ in data_loaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels)
        test_acc = running_corrects.double() / dataset_sizes['test']

    print("Test set acc: {:.4f}".format(test_acc))

    with open("test_acc_master.txt", 'a') as f:
        f.write("{},".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SpectralNet Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num_bands', '-n', nargs='*', default=2, type=int, help='number of input spectral bands')
    args = parser.parse_args()

    # select desired spectral bands
    means = [0.0145, 0.0198, 0.0246, 0.0952, 0.1538, 0.1721, 0.1681, 0.1754, 0.1829, 0.1707]
    stds = [0.0137, 0.017, 0.02, 0.06465, 0.096, 0.1075, 0.1062, 0.11044, 0.1146, 0.1084]

    cur_means = []
    cur_stds = []
    for b in args.num_bands:
        # rasterio indexing starts at 1 and that is what the flag corresponds to 
        cur_means.append(means[int(b)-1])
        cur_stds.append(stds[int(b)-1])
    print(args.num_bands)
    print(cur_means)
    print(cur_stds)

    writer = SummaryWriter()

    # lambda function to resize array
    #set_size = lambda x : resize(x, (64,64), preserve_range=True)
    #set_size = lambda x : imresize(x, (64,64))
    data_transforms = {
        'train': transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(cur_means, cur_stds),
                                    ]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(cur_means, cur_stds),
                                  ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(cur_means, cur_stds),
                                  ])
    }

    #data_dir = '/Users/senecal/Repos/hyperspectral/data/Tomato2'
    data_dir = 'D:\Repos\intel_spectrumnet\data\Tomato2'

    image_datasets = {x: DatasetFolder(os.path.join(data_dir, x), ['.tiff'], num_bands=args.num_bands, transform=data_transforms[x]) for x in ['train']} #, 'val', 'test']}
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train']} #, 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']} #, 'val', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get network and savepath 
    net, filename = getNetwork(len(args.num_bands))
    net.to(device)

    # set up loss criterion and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #optimizer = optim.RMSprop(net.parameters(), lr=0.001)

    # lr scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # start training 
    model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=50)

    # run best model on test set
    test_model(model_ft)

    # save best model 
    print("Saving best model...")
    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    save_point = os.path.join("checkpoint", filename)
    torch.save(model_ft.state_dict(), save_point + '.pt')
    print("Model saved!")

