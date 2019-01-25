import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch 

from tensorboardX import SummaryWriter

import os 
import sys 
import time 
import argparse
import datetime
import copy 

from spectrumnet import SpectrumNet
from GeoTiffDataset import DatasetFolder


def get_pretrained_network(file, num_bands, num_output_classes, version=1.0):
    net = SpectrumNet(num_bands=num_bands, version=version)
    net.load_state_dict(torch.load(file, map_location='cuda:0'))

    # for param in net.parameters():
    #     param.requires_grad = False

    # change the last conv2d layer
    net.classifier._modules["1"] = nn.Conv2d(512, num_output_classes, kernel_size=1)
    # change the internal num_classes variable rather than redefining the forward pass
    net.num_classes = num_output_classes

    tm = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = 'SpectrumNet-'+str(num_bands)+tm 

    return net, file_name


# Return network and filename 
def getNetwork(num_bands):
    net = SpectrumNet(version=1.0, num_bands=num_bands)
    tm = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = 'SpectralNet-'+str(num_bands)+tm 
    return net, file_name


# Training 
def train_model(criterion, num_epochs=25, num_bands=3):
    since = time.time() 
    input_num = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for fold in range(10):
        # get network and savepath 
        if(args.resume):
            print("Resuming from checkpoint...")
            #fname = '/media/jsen/SanDisk/Repos/intel_spectrumnet/data/SpectrumNet-102019-01-11_13:33:28.pt'
            #fname = '/media/jsen/SanDisk/Repos/intel_spectrumnet/data/SpectrumNet_DWS.pt'
            fname = 'pretrained_models/Spectrum_DWS2.pt'
            model, f = get_pretrained_network(file=fname,
                                              num_bands=len(args.num_bands),
                                              num_output_classes=2,
                                              version=1.1)
        else:
            model, f = getNetwork(len(args.num_bands))
        model.to(device)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0 
        
        # set up optimizer 
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,  nesterov=True) #weight_decay=5e-4

        # lr scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            print("-" * 10)

            # Train on 9 folds test on 1
            for phase in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                if phase != str(fold):
                    model.train() # Set model to training mode 
                else:
                    if scheduler is not None:
                        scheduler.step()
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
                    with torch.set_grad_enabled(phase != str(fold)):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backprop only if training phase 
                        if phase != str(fold):
                            loss.backward()
                            optimizer.step() 
                            writer.add_scalar('data/train_loss', loss.item(), input_num)
                            input_num += 1

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                if phase == str(fold):
                    writer.add_scalar("data/acc", epoch_acc, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model 
                if phase == str(fold) and epoch_acc > best_acc: 
                    print("Test fold: {} acc: {}".format(fold, epoch_acc))
                    best_acc = epoch_acc 
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            print() 
        
        time_elapsed = time.time() - since 
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best validation Acc: {:.4f}".format(best_acc))

        with open("crossval_acc_3band.txt", 'a') as f:
            f.write("{},".format(best_acc))

        # load best model weights 
        model.load_state_dict(best_model_wts)
    return model 

def test_model(model):
    running_corrects = 0

    model.eval()
    for inputs, labels in data_loaders['test']:
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
    for i in range(len(args.num_bands)):
        # rasterio indexing starts at 1 and that is what the flag corresponds to 
        cur_means.append(means[i])
        cur_stds.append(stds[i])
    print(args.num_bands)
    print(cur_means)
    print(cur_stds)

    writer = SummaryWriter()

    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cur_means, cur_stds)])

    data_dir = '/media/jsen/SanDisk/Repos/intel_spectrumnet/data/Avocado_2Class_CrossVal'

    # create datasets
    folds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    image_datasets = {x: DatasetFolder(os.path.join(data_dir, x), ['.tiff'], num_bands=args.num_bands, transform=data_transforms) for x in folds}
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in folds}

    dataset_sizes = {x: len(image_datasets[x]) for x in folds}
    class_names = image_datasets['0'].classes

    # set up loss criterion 
    criterion = nn.CrossEntropyLoss()

    # start training 
    model_ft = train_model(criterion, num_epochs=50, num_bands=len(args.num_bands))


