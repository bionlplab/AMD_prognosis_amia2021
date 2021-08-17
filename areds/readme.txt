Run main.py to
  extract wanted features from the AREDS AMD dbGaP-25378-phenotype Dataset
  format in a python readable dataset
  train multi-task learning frameworks using EfficientNet-B3 or ResNet152 architectures trained on Drusen Size and Pigmentation Abnormality
  extract vector representations of each image from visits 00, 04, 06
  Train LSTM, MLP, and CoxPH survival models over a range of learning rates or penalizers and evaluate on the development set
  Find the best learning rate or penalizer weight based on development set for each model type, type of data used, with or without including historical visit data
  Evaluate each model using best respective learning rate or penalizer on test set 
  

Folder setup:
  areds <-- main folder
  areds/data <-- where the data gets written to and read from. Must include dbGaP-25378-phenotype
  areds/ImageFolder/PatientID <-- includes the fundus photographs of that patient from visits 00, 04, 06
  areds/Image_FolderAll/PatientID <-- includes all of the fundus photographs of that patient
