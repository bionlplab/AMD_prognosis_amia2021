import numpy as np
import h5py
import pandas as pd
import matplotlib.image as mpimg
import os.path
import torch
from PIL import Image
from torchvision import transforms
import hdfdict
import os
import torchvision.models as models
from torch import nn



class MTL_Model(nn.Module):
    def __init__(self, mod, fc_size ):
        super(MTL_Model, self).__init__()
        self.modtype = mod
        self.fc_size = fc_size
    
        self.MTL1 = nn.Linear(self.fc_size,3) #should we add sigmoid?
        
        self.MTL2 = nn.Linear(self.fc_size,2)

    def forward(self, x):
        x1 = self.modtype(x)
        x2 = self.MTL1(x1)
        x3 = self.MTL2(x1)

        return x2.requires_grad_(True), x3.requires_grad_(True)

def Extract_Image_Features(model_type = 'resnet_fine')
    '''
        Extracts deep image features using multi-task learning model trained on pigmentation abnormalities and drusen size

        arguments:
            model_type - One of ['Efficientnet_fine', 'Resnet_fine', 'Resnet_pretrained','Efficientnet_pretrained']

        returns:
            none
            exports feature set to current_working_directory/Extracted_Image_Features/model_type.pth
    '''
    load_folder = model_type
    save_new_h5_as = load_folder.strip('/')+'.h5'
    print(save_new_h5_as)
    if ('resnet' in model_type.lower()):
        model_RN = models.resnet152(pretrained=True)
        model_RN.fc = nn.Identity()
        model_fc_size = 2048
    elif ('efficient' in model_type.lower()):
        model_RN = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
        model_RN.classifier = nn.Identity()
        model_fc_size = 1536
    else:
        print('model_type must be either resnet or efficientnet, it was: ' + model_type)
    model = MTL_Model(model_RN, model_fc_size)
    model = model.float()
    model.load_state_dict(torch.load('Trained_Models/'+load_folder+'/Best_model_test.pth'))
    model.MTL1 = nn.Identity()
    model.MTL2 = nn.Identity()
    model.eval()

    path = '/home/gcg4001/AMD_Data/AMD_Data/'

    split = 200000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model=model.to(device)


    #Get list of original images used
    file_name = 'areds1_resnetv2.h5'
    all_data =  h5py.File('data/' + file_name, 'r')
    key_list = list(all_data.keys())
    patients = np.unique([int(i.split(' ')[0]) for i in key_list])
    #patients = patients[:10]
    key_list = [i for i in key_list if int(i.split(' ')[0]) in patients]
    del all_data
    #load in all images

    images = {}
    no_image = []
    for i in key_list:
        if(os.path.isfile('/home/gcg4001/ImageFolder/'+i.split(' ')[0]+'/' + i + '.jpg')):
     #       images[i] = mpimg.imread(str('/home/gcg4001/ImageFolder/'+i.split(' ')[0]+'/' + i + '.jpg'))
            print('FOUND: /home/gcg4001/ImageFolder/'+i.split(' ')[0]+'/' + i + '.jpg')
        else:
            no_image.append(i)
    for i in no_image:
        key_list.remove(i)

    print('Could not find these images: ')
    for i in no_image:
        print(i)
    print('Missing ' + str(len(no_image)) +' images')

    patient_set = set()
    for i in key_list:
        patient_set.add(i.split(' ')[0])
    print(str(len(key_list)) + ' images found for '+str(len(patient_set)) + ' patients.')

    import time
    start = time.time()
    outputs = {}
    major_count = 0

    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      transforms.Normalize(mean=[99.00661305/255, 54.5993171/255, 28.87143279/255], std=[85.23130083/255, 48.28956532/255, 25.62829875/255])
    ])

    for count, i in enumerate(key_list):
        input_image = Image.open(str('/home/gcg4001/ImageFolder/'+i.split(' ')[0]+'/' + i + '.jpg'))
        #print('Reducing: /home/gcg4001/ImageFolder/'+i.split(' ')[0]+'/' + i + '.jpg')
        print(str(count)+'/'+str(len(key_list)), end=':  ')
        print(str(time.time()-start)+'/'+str((time.time()-start)*len(key_list)/(count+1)))
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch=input_batch.to(device)

        with torch.no_grad():
            temp = model(input_batch)
            outputs[i] = temp[0].cpu().detach().numpy()
            #print('Number of keys: ' + str(len(outputs.keys())))
        if(i==1):
            print(outputs[i].shape)

        if((count+1)%split == 0):
            if(os.path.isfile( path + 'h5_files/'+str(major_count)+save_new_h5_as)):
                os.remove(path + 'h5_files/'+str(major_count)+ save_new_h5_as)
            hdfdict.dump(outputs, path + 'h5_files/'+str(major_count)+ save_new_h5_as)
            outputs = {}
            major_count = major_count + 1

    if not os.path.isdir('Extracted_Image_Features'):
        os.mkdir('Extracted_Image_Features')
    save_file = os.getcwd() + '/Extracted_Image_Features/' save_new_h5_as
    print('Exported extracted image features to ' + save_file)
    print(str(len(outputs)) + ' images for ' +str(len(patient_set))+' patients')

    if(os.path.isfile( save_file)):
        os.remove(save_file)
    hdfdict.dump(outputs, save_file)

