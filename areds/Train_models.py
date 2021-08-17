import pandas
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchtuples as tt
from torchvision import transforms
from torch import nn
import torchvision.models as models
import numpy as np
import random
import torch
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import copy
from datetime import date
import imgaug.augmenters as iaa



class MTL_Model(nn.Module):
    def __init__(self, mod, fc_size ):
        super(MTL_Model, self).__init__()
        self.modtype = mod
        self.fc_size = fc_size
    
        self.MTL1 = nn.Linear(self.fc_size,3) #Drusen Size
        
        self.MTL2 = nn.Linear(self.fc_size,2) #Pigmentation Abnormality
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        x1 = self.modtype(x)
        x1 = self.dropout(x1)
        #print(x1.shape)
        x2 = self.MTL1(x1)
        x3 = self.MTL2(x1)

        return x2.requires_grad_(True), x3.requires_grad_(True)


class AMD_image_loader(Dataset):
    def __init__(self, image_folder_path = '/home/gcg4001/Image_FolderAll/', clinical_folder_path = 
                '/home/gcg4001/AMD_Data/'):
        self.image_folder_path = image_folder_path
        self.clinical_folder_path = clinical_folder_path
        self.clinical = pandas.read_csv(self.clinical_folder_path + 'patient_info_fin.csv', index_col=0)[['VISNO','LE_IMG', 'RE_IMG', 'drusen_le', 'drusen_re', 'pig_le', 'pig_re']]
        self.clinical = self.clinical[self.clinical['VISNO']>6] #Only take visits >6 for training
        #self.clinical = self.clinical[self.clinical.isna().any(axis=1)==False]
        self.clinical = self.clinical[(self.clinical==88).any(axis=1)==False] #remove visits with missing data
        self.images = []
        for i in range(self.clinical.shape[0]):
            LE_IMG = self.clinical.iloc[i]['LE_IMG']
            fname = image_folder_path + str(LE_IMG).split(' ')[0]+'/'+str(LE_IMG)
            if(os.path.isfile(fname)):
                self.images.append(LE_IMG)

            RE_IMG = self.clinical.iloc[i]['RE_IMG']
            fname = image_folder_path + str(RE_IMG).split(' ')[0]+'/'+str(RE_IMG)
            if(os.path.isfile(fname)):
                self.images.append(RE_IMG)
        #len(self.images))
        self.set_type = 'test'
        self.preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[99.00661305/255, 54.5993171/255, 28.87143279/255], std=[85.23130083/255, 48.28956532/255, 25.62829875/255])
    ])
        self.augment = transforms.Compose([
            iaa.Sequential([
                iaa.Fliplr(p=0.5),
                iaa.Crop(percent=(0, 0.2)), # random crops
                #But we only blur about 50% of all images.
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.25))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.85, 1.15)),
#                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                iaa.Affine(
                        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                        rotate=(-5, 5),
                        shear=(-2, 2)
                    )
            ]).augment_image])


    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        fname = self.images[idx]
        image_path = self.image_folder_path + str(fname).split(' ')[0]+'/'+str(fname)
        return_image = Image.open(image_path)
        
        if(' LE ' in fname):
            return_clinical = self.clinical[self.clinical['LE_IMG'] == fname][['drusen_le', 'pig_le']]
        elif(' RE ' in fname):
            return_clinical = self.clinical[self.clinical['RE_IMG'] == fname][['drusen_re', 'pig_re']]
        else:
            print('ERROR on image ' + fname)    
        return_clinical = tt.tuplefy(np.array(return_clinical))
        if(self.set_type == 'train'):
            return_image = self.preprocess(Image.fromarray(self.augment(np.array(return_image)))).float()
        elif(self.set_type == 'test'):
            return_image = self.preprocess(return_image).float()
        else:
            print("ERROR: set_type is " + self.set_type + ' and should be \'train\' or \'test\'')
        
        return return_image, torch.tensor(return_clinical[0]).long()
      

def seed_everything(seed=1234):
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False




def train_MTL_model(model_type = 'resnet_fine',image_folder_path = '/home/gcg4001/Image_FolderAll/', clinical_folder_path =
                '/home/gcg4001/AMD_Data/'):
    '''
        Trains multi-task learning model using images from after visit 06 of all patients trained on presence of pigmentation abnormalities and drusen size

        arguments:
            model_type - One of ['Efficientnet_fine', 'Resnet_fine', 'Resnet_pretrained','Efficientnet_pretrained']
            image_folder_path - path of folder containing all FUNDUS photographs
            clinical_folder_path - path to patient_info_fin.csv, generated from clean_json()

        returns:
            none
            exports trained MTL model to current_working_directory/Trained_MTL_Models/model_type/Best_model_test.pth
    '''
    today = date.today()
    d4 = today.strftime("%b_%d_%Y")

    seed = 1234
    LR = 0.0005
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU COUNT AND NAME: ')
    print(torch.cuda.device_count())
    if (torch.cuda.device_count() > 0):
        torch.cuda.set_device(0)
        print(torch.cuda.current_device())

    print('loading data')
    dataset = AMD_image_loader(image_folder_path = image_folder_path, clinical_folder_path = clinical_folder_path)
    training_set, test_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)),
                                                                     len(dataset) - int(0.9 * len(dataset))],
                                                           generator=torch.Generator().manual_seed(seed))

    batch_size = 16

    dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=0)  # , collate_fn=collate_fn)
    dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  num_workers=0)  # , collate_fn=collate_fn)
    name = model_type
    if('resnet' in model_type.lower()):
        model_RN = models.resnet152(pretrained=True)
        model_RN.fc = nn.Identity()
        model_fc_size = 2048
    elif('efficient' in model_type.lower()):
        model_RN = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
        model_RN.classifier = nn.Identity()
        model_fc_size = 1536
    else:
        print('model_type must be either resnet or efficientnet, it was: '+model_type)

    for parameters in model_RN.parameters():
        parameters.requires_grad = False



    epochs = 10
    if('pretrain' in name):
        epochs = 1
    train_loss = []
    test_loss = []
    test_metrics = []
    best_loss = 2000
    best_loss_test = 2000
    best_model_wts = 0

    save_folder = 'Trained_MTL_Models/'+name+'/'
    if not os.path.isdir('Trained_MTL_Models'):
        os.mkdir('Trained_MTL_Models')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    print(save_folder)
    readme = ['']
    readme[0] = str(dataset.preprocess)

    readme.append(str(dataset.augment.transforms[0]))
    pandas.DataFrame(readme).to_csv(save_folder+'readme.csv')


    model = MTL_Model(model_RN, model_fc_size)
    model = model.float()
    model = model.to(device)
    if(device =='cuda'):
        model = model.cuda()
    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr= LR)
    all_losses = []
    res = []

    print('Calculating weights')
    dataset.set_type = 'train'
    for i in dataloader_train:
        data, label = i

    drusen_counter = collections.Counter(np.array(label[:,0,0]))
    loss_weight_drusen = np.array([drusen_counter[0],drusen_counter[1],drusen_counter[2]])
    loss_weight_drusen = 1 - loss_weight_drusen/np.sum(loss_weight_drusen)
    loss_weight_drusen = loss_weight_drusen/np.sum(loss_weight_drusen)
    loss_weight_drusen = torch.tensor(loss_weight_drusen)
    pig_counter = collections.Counter(np.array(label[:,0,1]))
    loss_weight_pig = [pig_counter[0],pig_counter[1]]
    loss_weight_pig = 1 - loss_weight_pig/np.sum(loss_weight_pig)
    loss_weight_pig = torch.tensor(loss_weight_pig)

    loss_drusen = nn.CrossEntropyLoss(weight=loss_weight_drusen.float())
    loss_pig = nn.CrossEntropyLoss(weight=loss_weight_pig.float())
    loss_drusenv = nn.CrossEntropyLoss(weight=loss_weight_drusen.float())
    loss_pigv = nn.CrossEntropyLoss(weight=loss_weight_pig.float())
    loss_drusen = loss_drusen.to(device)
    loss_pig = loss_pig.to(device)
    loss_drusenv = loss_drusenv.to(device)
    loss_pigv = loss_pigv.to(device)
    print('Starting training')
    for epoch in tqdm(range(epochs)):
        loss_in_epoch = []
        loss_in_epoch_test = []
        dataset.set_type = 'train'
        for n, i in enumerate(dataloader_train):
            model = model.train()
            if ('pretrain' in name):
                model = model.eval()
            data, label = i
            #print(data.shape)
            optimizer.zero_grad()
            data = data.to(device)
            output_drusen, output_pig = model(data)

            #output_drusen = output_drusen.to(device)
            #output_pig = output_pig.to(device)
            label = label.to(device)
            loss_drusen_value = loss_drusen(output_drusen, label[:,0,0])
            loss_pig_value = loss_pig(output_pig, label[:,0,1])
            loss = (loss_drusen_value + loss_pig_value)/2
            if ('pretrain' not in name):
                loss.backward()
                optimizer.step()
            loss_in_epoch.append(loss.item())
        epoch_loss = np.mean(loss_in_epoch)

        if(epoch_loss < best_loss):
            print('Saving best loss')
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model.to('cpu').state_dict(),save_folder+'Best_model.pth')
            model.to(device)

        dataset.set_type = 'test'
        with torch.no_grad():
            for iv in dataloader_test:
                model = model.eval()
                datav, labelv = iv
                datav = datav.to(device)
            #    print(datav.shape)

                optimizer.zero_grad()

                output_drusenv, output_pigv = model(datav)
                labelv = labelv.to(device)
                loss_drusenv_value = loss_drusenv(output_drusenv, labelv[:,0,0])
                loss_pigv_value = loss_pigv(output_pigv, labelv[:,0,1])
                lossv = (loss_drusenv_value + loss_pigv_value)/2
                loss_in_epoch_test.append(lossv.item())

            epoch_loss_test = np.mean(loss_in_epoch_test)

            if(epoch_loss_test < best_loss_test):
                print('Saving best loss - Test')
                best_loss_test = epoch_loss_test
                best_model_wts_test = copy.deepcopy(model.state_dict())
                torch.save(model.to('cpu').state_dict(),save_folder+'Best_model_test.pth')
                model.to(device)

        all_losses.append([epoch_loss, epoch_loss_test])
        pandas.DataFrame(all_losses, columns = ['mean_loss_in_epoch_train', 'mean_loss_in_epoch_test']).to_csv(save_folder+'Losses.csv')
        res.extend([f'train_loss: {round(epoch_loss,4)},  test_loss: {round(epoch_loss_test,4)}'])
        print(res)
        print(save_folder)
        if(best_loss < (0.8*best_loss_test) and epoch > 4):
            break
      
