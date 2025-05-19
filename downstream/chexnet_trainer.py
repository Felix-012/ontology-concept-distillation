import os
import numpy as np
import time
import sys
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import pandas as pd

from data_utils import map_to_label_vector
from downstream.models import DenseNet121
from downstream.models import DenseNet169
from downstream.models import DenseNet201
from downstream.dataset import DatasetGenerator


#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathDirTrainData, pathFileTrain, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint, model_save_path, centercrop_only, use_cache, debug):

        # Initialize lists to store losses
        train_losses = []
        val_losses = []
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        # Load and split the CSV file
        df = pd.read_csv(pathFileTrain)
        file_paths = df.path.values.tolist()
        findings = df["Finding Labels"]#df.findings.values.tolist()
        labels = [*map(lambda x: map_to_label_vector(x, nnClassCount), findings)]
        
        # Randomly split into train/val with fixed seed
        generator = torch.Generator().manual_seed(42)
        num_samples = len(file_paths)
        indices = torch.randperm(num_samples, generator=generator).tolist()

        if debug: 
            print("Starting debug mode with limited data!")
            indices = indices[:(trBatchSize * 5)]
            num_samples = len(indices)

        split = int(np.floor(0.8 * num_samples))
        train_idx, val_idx = indices[:split], indices[split:]
        
        train_files = [file_paths[i] for i in train_idx]
        val_files = [file_paths[i] for i in val_idx]

        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        datasetTrain = DatasetGenerator(pathDirTrainData, train_files, train_labels, transform=transformSequence, use_cache=use_cache)
        datasetVal = DatasetGenerator(pathDirTrainData, val_files, val_labels, transform=transformSequence)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=16, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        print(f"Saving results to {model_save_path}")
        #---- TRAIN THE NETWORK
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            lossTrain = ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            # Store losses
            train_losses.append(lossTrain)
            val_losses.append(lossVal)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()},model_save_path)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
            # Plot and save training curves
            plot_save_path = os.path.join(os.path.dirname(model_save_path), 'training_curves.png')
            plot_training_curves(train_losses, val_losses, plot_save_path)

        print(f"Training curves saved to {plot_save_path}")
        print(f"Models saved to {model_save_path}")
    
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        # Initialize tqdm progress bar
        with tqdm(total=len(dataLoader), desc="Training", unit="batch") as pbar:
            running_loss = 0.0  # To accumulate loss
            for batchID, (input, target) in enumerate(dataLoader, 1):  # Start batchID from 1
                target = target.cuda()
                
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)

                lossvalue = loss(varOutput, varTarget)

                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += lossvalue.item()

                # Log average loss every 10 steps
                if batchID % 10 == 0:
                    avg_loss = running_loss / 10
                    pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
                    running_loss = 0.0  # Reset running loss

                # Update the progress bar
                pbar.update(1)

            # Handle the case where the last few batches are not a multiple of 10
            if running_loss > 0:
                avg_loss = running_loss / (batchID % 10)
                pbar.set_postfix({"Avg Loss (final)": f"{avg_loss:.4f}"})
        return avg_loss
                
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0

        with torch.no_grad():        
            for i, (input, target) in enumerate (dataLoader):
                
                target = target.cuda()
                    
                #varInput = torch.autograd.Variable(input, volatile=True)
                #varTarget = torch.autograd.Variable(target, volatile=True)    
                output = model(input)
                
                losstensor = loss(output, target)
                losstensorMean += losstensor
                
                lossVal += losstensor.item()
                lossValNorm += 1
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp, outfile):   
        #CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        #        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        from data_utils import CLASSES
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        # load filepaths and labels from csv
        df = pd.read_csv(pathFileTest)
        file_paths = df.path.values.tolist()
        findings = df["Finding Labels"]#df.findings.values.tolist()
        labels = [*map(lambda x: map_to_label_vector(x, nnClassCount), findings)]

        datasetTest = DatasetGenerator(pathDirData, file_paths, labels, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        print("Starting Evaluation")
        model.eval()
        
        with tqdm(total=len(dataLoaderTest), desc="Testing", unit="batch") as pbar:
            with torch.no_grad():
                for i, (input, target) in enumerate(dataLoaderTest):
                    
                    target = target.cuda()
                    outGT = torch.cat((outGT, target), 0)
                    
                    bs, n_crops, c, h, w = input.size()
                    
                    varInput = input.view(-1, c, h, w).cuda()
                    
                    out = model(varInput)
                    outMean = out.view(bs, n_crops, -1).mean(1)
                    
                    outPRED = torch.cat((outPRED, outMean.data), 0)
                    pbar.update(1)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASSES[i], ' ', aurocIndividual[i])
        

        with open(outfile, "w") as f:
            f.write(f"AUROC mean: {aurocMean}\n")
            for i in range(len(aurocIndividual)):
                f.write(f"{CLASSES[i]}: {aurocIndividual[i]}\n")

        print(f"Results saved to {outfile}")
     
        return

def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves and save the plot.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
