from Clean_Json import clean_json
from Find_Best_Hyperparameters import get_best_lr
import numpy as np
from Pycox import evaluate_model
from Train_models import train_MTL_model
from Make_h5_trained_models.py import Extract_Image_Features
import warnings
import pandas
import os
from extract_variables import extract_variables

extract_variables()
clean_json()
for Clinical_Set in ['Efficientnet_fine', 'Resnet_fine', 'Resnet_pretrained','Efficientnet_pretrained']:
    train_MTL_model(Clinical_Set)
    Extract_Image_Features(Clinical_Set)

LRs = 1 / (10 ** (np.arange(4, 15) / 4))
visits = [['00', '04', '06'],['00']]
Clinical_Sets = ['a','b','Efficientnet_fine', 'Resnet_fine', 'Resnet_pretrained','Efficientnet_pretrained']
best_lrs = {}
best_lrs_df = pandas.DataFrame()
model_types = ['LSTM','MLP','CoxPH'] #
for model_type in model_types:
    for visit in visits:
        for Clinical_Set in Clinical_Sets:
            if(model_type == 'LSTM' and len(visit) == 1):
                best_lrs[Clinical_Set.upper()+str(len(visit))] = 0
            else:
                results = evaluate_model(top_dir = os.getcwd()+'/data/', visits = visit, Clinical_Set = Clinical_Set, model_type=model_type, test='dev', LRs = LRs)
                best_lr = get_best_lr(results)
                best_lrs[Clinical_Set.upper()+str(len(visit))] = best_lr
            print(best_lrs)
    best_lrs_df = pandas.concat([best_lrs_df, pandas.DataFrame.from_dict(best_lrs, orient='index', columns = [model_type])], axis=1)
    best_lrs_df.to_csv('data/Best_LRs.csv')

best_lrs_df = pandas.read_csv('data/Best_LRs.csv', index_col=0)
test_results = pandas.DataFrame()
for model_type in model_types:
    for visit in visits:
        for Clinical_Set in Clinical_Sets:
            if(model_type == 'LSTM' and len(visit) == 1):
                pass
            else:
                lr = [best_lrs_df.loc[Clinical_Set.upper()+str(len(visit))][model_type]]
                results = evaluate_model(top_dir = '/Users/gregory/Documents/Weill_Cornell/Wang_Lab/AMD/AMIA/Coding_Files/data/', visits = visit, Clinical_Set = Clinical_Set, model_type=model_type, test='test', LRs = lr)
                #results.to_csv(model_type+'_'+Clinical_Set.upper()+str(len(visit))+'.csv')
                results.columns = [model_type+'_'+Clinical_Set.upper()+str(len(visit))]
                test_results = pandas.concat([test_results, results], axis=1)
            test_results.to_csv('TEST_RESULTS.csv')
                

