import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

from datetime import datetime
import os
import glob
import copy
import sys

from workspace_utils import active_session

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_names = ['densenet121', 'densenet161', 'resnet18', 'vgg16']
datadir = 'flowers'
savedir = 'chksav'

# main
def main():
    # get input arguments and print
    args = get_input_args()
    print('\n*** command line arguments ***')
    print('architecture:', args.arch, '\ndata dir:', args.data_dir, '\nchkpt dir:', args.save_dir,
          '\nlearning rate:', args.learning_rate, '\ndropout:', args.dropout,
          '\nhidden layer:', args.hidden_units, '\nepochs:', args.epochs, '\nGPU mode:', args.gpu, '\n')

    if len(glob.glob(args.data_dir)) == 0:
        print('*** data dir: ', args.data_dir, ', not found ... exiting\n')
        sys.exit(1)

    if args.learning_rate <= 0:
        print('*** learning rate cannot be negative or 0 ... exiting\n')
        sys.exit(1)

    if args.dropout < 0:
        print('*** dropout cannot be negative ... exiting\n')
        sys.exit(1)

    # if arch is not resnet18 and hidden units supplied, check values are numeric
    if args.arch != 'resnet18':
        if args.hidden_units:
            try:
                list(map(int, args.hidden_units.split(',')))
            except ValueError:
                print("hidden units contain non numeric value(s) :[", args.hidden_units, "], ... exiting\n")
                sys.exit(1)

    if args.epochs < 1:
        print('*** epochs cannot be less than 1 ... exiting\n')
        sys.exit(1)

    # transform and load training, validatation and testing sets
    dataloaders, image_datasets = transform_load(args)

    # load pre-trained model and replace with custom classifier
    model = models.__dict__[args.arch](pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    if args.arch == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, len(dataloaders['train'].dataset.classes))
        print('\n*** model architecture:', args.arch,'\n*** fc:\n', model.fc, '\n')
        # set training criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        model = build_classifier(model, args, dataloaders)
        print('\n*** model architecture:', args.arch,'\n*** Classifier:\n', model.classifier, '\n')
        # set training criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    # start model training and testing
    if device.type == 'cuda':
        if args.gpu:
            print('*** GPU is available, using GPU ...\n')
        else:
            print('*** training model in GPU mode ...\n')
    else:
        if args.gpu:
            print('*** GPU is unavailable, using CPU ...\n')
        else:
            print('*** training model in CPU mode ...\n')

    with active_session():
        model = train(model, dataloaders, optimizer, criterion, args.epochs, 40, args.learning_rate)
        model = test(model, dataloaders, criterion, args.arch)

    # save to checkpoint
    model = model.cpu() # back to CPU mode post training
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx

    # if checkpoint dir not exists, create it
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': args.arch,
        'lrate': args.learning_rate,
        'epochs': args.epochs}

    if args.arch == 'resnet18':
        checkpoint['fc'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier

    chkpt = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.arch + '.pth'
    checkpt = os.path.join(args.save_dir, chkpt)

    torch.save(checkpoint, checkpt)
    print('\n*** checkpoint: ', chkpt, ', saved to: ', os.path.dirname(checkpt), '\n')


def get_input_args():
    # create parser
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, nargs='?', default=datadir,
                        help='datasets path')

    parser.add_argument('--save_dir', type=str, default=savedir,
                        help='checkpoint directory')

    parser.add_argument('--arch', dest='arch', default='densenet121',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: densenet121)')

    parser.add_argument('-lr','--learning_rate', dest='learning_rate', default=0.001, type=float,
                        help='Network learning rate (default: 0.001)')

    parser.add_argument('-dout','--dropout', dest='dropout', default=0.5, type=float,
                        help='hiddden layer dropout (default: 0.5)')

    parser.add_argument('-hu','--hidden_units', dest='hidden_units', default=None, type=str,
                        help='hidden units, one or multiple values (comma separated) ' +
                        """ enclosed in single quotes. Ex1. one value: '512'
                            Ex2. multiple values: '1000, 512' """)

    parser.add_argument('-e','--epochs', dest='epochs', default=5, type=int,
                        help='Number of Epochs (default: 5)')

    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help='train in gpu mode')

    return parser.parse_args()

def transform_load(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    # define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        }

    # define datasets
    image_datasets = {k: datasets.ImageFolder(os.path.join(args.data_dir, k), transform=data_transforms[k])
                      for k in ['train','valid','test']}

    #  define dataloaders
    dataloaders = {k: torch.utils.data.DataLoader(image_datasets[k], batch_size=32, shuffle=True)
                   for k in ['train','valid','test']}

    return dataloaders, image_datasets

def build_classifier(model, args, dataloaders):
    in_size = {
        'densenet121': 1024,
        'densenet161': 2208,
        'vgg16': 25088,
        }

    hid_size = {
        'densenet121': [480],
        'densenet161': [1000, 500],
        'vgg16': [4096, 4096,1000],
        }

    output_size = len(dataloaders['train'].dataset.classes)
    relu = nn.ReLU()
    dropout = nn.Dropout(args.dropout)
    output = nn.LogSoftmax(dim=1)

    if args.hidden_units:
        h_list = args.hidden_units.split(',')
        h_list = list(map(int, h_list)) # convert list from string to int
    else:
        h_list = hid_size[args.arch]

    h_layers = [nn.Linear(in_size[args.arch], h_list[0])]
    h_layers.append(relu)
    if args.arch[:3] == 'vgg':
        h_layers.append(dropout)

    if len(h_list) > 1:
        h_sz = zip(h_list[:-1], h_list[1:])
        for h1,h2 in h_sz:
            h_layers.append(nn.Linear(h1, h2))
            h_layers.append(relu)
            if args.arch[:3] == 'vgg':
                h_layers.append(dropout)

    last = nn.Linear(h_list[-1], output_size)
    h_layers.append(last)
    h_layers.append(output)

    print(h_layers)
    model.classifier = nn.Sequential(*h_layers)

    return model

# validate model
def validate(model, dataloaders, criterion):
    valid_loss = 0
    accuracy = 0

    for images, labels in iter(dataloaders['valid']):

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


# train model for densenet121, densenet161 and vgg16
def train(model, dataloaders, optimizer, criterion, epochs=2, print_freq=20, lr=0.001):

    model.to(device)
    start_time = datetime.now()

    print('epochs:', epochs, ', print_freq:', print_freq, ', lr:', lr, '\n')

    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(dataloaders['train']):
            steps +=1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_freq == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, dataloaders, criterion)

                print('Epoch: {}/{}..'.format(e+1, epochs),
                      'Training Loss: {:.3f}..'.format(running_loss/print_freq),
                      'Validation Loss: {:.3f}..'.format(valid_loss/len(dataloaders['valid'])),
                      'Validation Accuracy: {:.3f}%'.format(accuracy/len(dataloaders['valid']) * 100)
                     )
                running_loss = 0

                model.train()

    elapsed = datetime.now() - start_time

    print('\n*** classifier training done ! \nElapsed time[hh:mm:ss.ms]: {}'.format(elapsed))
    return model

# test model
def test(model, dataloaders, criterion, arch):
    print('\n*** validating testset ...\n')
    model.cpu()
    model.eval()

    test_loss = 0
    total = 0
    match = 0

    start_time = datetime.now()

    with torch.no_grad():
        for images, labels in iter(dataloaders['test']):

            model, images, labels = model.to(device), images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            total += images.shape[0]
            equality = labels.data == torch.max(output, 1)[1]
            match += equality.sum().item()

    model.test_accuracy = match/total * 100
    print('Test Loss: {:.3f}'.format(test_loss/len(dataloaders['test'])),
            'Test Accuracy: {:.2f}%'.format(model.test_accuracy))

    time_taken = datetime.now() - start_time
    print('\n*** test validation done ! \n\nTime Taken [hh:mm:ss.ms] {}:'.format(time_taken))
    return model

# Call to main function to run the program
if __name__ == "__main__":
    main()