#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torchvision
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from PIL import ImageFile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Adapted code from lesson3, ex13 finetune_a_cnn_solution.py and lesson4, ex13 cifar.py
def test(model, test_loader, criterion, device):    
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
 
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects/ len(test_loader)
     
    logger.info('Testing Accuracy: {:.2f}, Testing Loss: {:.2f}'.format(total_acc, total_loss))
                                                                        
        
        
def train(model, train_loader, validation_loader, criterion, optimizer, device):
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs = 30
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
          
            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
                    
            logger.info('Accuracy: {:.2f}, Loss: {:.2f}, Best loss {:.2f}'.format(epoch_acc, epoch_loss, best_loss))
            
        if loss_counter==1:
            break
    
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)    #see notes below
    num_classes = 133    #number of dog breeds
    
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_classes))
          
    return model

# Notes: Used resnet50 because it has the best accuracy and speed, and a small model size [https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/ , Accessed 11/12/21]
    
    
    

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
    #transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.ImageFolder(root= os.path.join(data, 'train'), transform=training_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        shuffle=True)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(data, 'test'), transform=testing_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    validset = torchvision.datasets.ImageFolder(root=os.path.join(data, 'valid'), transform=testing_transform)
        
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
        shuffle=False)
  
    return trainloader, testloader, validloader

    

def main(args):
    
   
    '''
    TODO: Initialize a model by calling the net function
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
  
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')
     
    logger.info("Initializing the model.")
    model=net()
    model=model.to(device)

    '''
    TODO: Create your loss and optimizer
    '''

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)  
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    logger.info("Loading data")
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    
    logger.info("Training the model.")
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device)

    
    '''
    TODO: Test the model to see its accuracy
    '''
   
    logger.info("Testing the model.")
    test(model, test_loader, loss_criterion, device)  
       
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth")) 

    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
      
    args=parser.parse_args()   
    
    main(args)
