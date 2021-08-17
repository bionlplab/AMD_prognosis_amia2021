import os
import pandas
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import math
import warnings
import statistics
from sklearn.metrics import roc_curve, auc, brier_score_loss
from scipy import stats


from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from lifelines import LogLogisticAFTFitter
from lifelines import LogNormalAFTFitter


COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rcParams['figure.facecolor'] = 'white'
#set_facecolor((0, 0,0))

def load_json(json_filename):
    data = pandas.read_json(json_filename)
    data['ID2'] = data['RC_ID']
    data.rename(columns = {'ID2': 'dbGaP SubjID'}, inplace = True)
    data = data.drop(columns = ['RC_ID'])
    data['dbGaP SubjID'].iloc[np.where((data['dbGaP SubjID']>1)==False)] = np.where((data['dbGaP SubjID']>1)==False)
    data['dbGaP SubjID'] = data['dbGaP SubjID'].astype(int)
    print(data.shape)

    return data

def rearrange_json(data, export_data=True, min_required_visits = 3, plot=False):
    visit0 = []
    for i in data['dbGaP SubjID']:
        pat = data[data['dbGaP SubjID'] == i]
        for j in pat.VISITS.values[0]:
            res = pandas.DataFrame(j, index = [i])
            visit0.append(res)
    fin_res = pandas.concat(visit0,0)

    res_combined = []
    for i in data['dbGaP SubjID']:
        patient1 = fin_res.loc[i]
        if patient1.ndim > 1:
            patient2 = data[data['dbGaP SubjID'] == i].iloc[:,1:]
            miss_col = np.setdiff1d(np.array(patient2.columns),np.array(patient1.columns))
            for j in miss_col:
                patient1[j] = patient2[j].values[0]
            res_combined.append(patient1)
    res_combined_fin = pandas.concat(res_combined,0)
    res_combined_fin = res_combined_fin.drop(columns = 'VISITS')

    risk_visit = pandas.DataFrame(np.nan, index=data['dbGaP SubjID'], columns=['VISNO_EVENT', 'EVENT'])
    for i in data['dbGaP SubjID']:
        pat = data[data['dbGaP SubjID'] == i]
        risk_visit.loc[i]['EVENT'] = 0
        for a in pat['VISITS'][pat['VISITS'].index[0]]:
            if ('SCALE' in a.keys()):
                risk_visit.loc[i]['VISNO_EVENT'] = a['VISNO']
                if (a['SCALE'] == 5):
                    risk_visit.loc[i]['VISNO_EVENT'] = a['VISNO']
                    risk_visit.loc[i]['EVENT'] = 1
                    break

    # remove image data
    total_combined = pandas.concat([risk_visit, res_combined_fin], axis=1, join='inner')
    image_columns = ['LE_IMG', 'RE_IMG', 'LE_IMG_DBGAP_FILE', 'RE_IMG_DBGAP_FILE']
    total_combined_image_files = total_combined[image_columns]
    total_combined_image_files['VISNO'] = total_combined['VISNO']
    total_combined_image_files.to_csv('image_files_per_visit.csv')
    total_combined = total_combined.drop(columns=image_columns)

    # remove patients with no 'SCALE' results
    no_visit_data = risk_visit.index[np.isnan(risk_visit['VISNO_EVENT'])]
    total_combined = total_combined.drop(list(no_visit_data))

    # Change 88 in lowercase columns to NaN
    col_names = total_combined.columns
    for col in col_names:
        if (col == col.lower()):
            total_combined.loc[total_combined[col] == 88, col] = np.nan

    # Change 8 in uppercase columns to NaN
    nan_8 = ['REDRSZWI', 'LEDRSZWI', 'REGEOAWI', 'LEGEOAWI', 'REGEOACT', 'LEGEOACT', 'REGEOACS', 'LEGEOACS', 'RERPEDWI',
             'LERPEDWI', 'REINCPWI', 'LEINCPWI', 'RESUBFF2', 'LESUBFF2', 'RENDRUF2', 'LENDRUF2', 'RESSRF2', 'LESSRF2',
             'RESUBHF2', 'LESUBHF2']
    for col in nan_8:
        total_combined.loc[total_combined[col] == 8, col] = np.nan

    total_combined = total_combined[~pandas.isnull(total_combined['SCALE'])]

    if(export_data==True):
        if (os.path.isdir('data') == False):
            os.mkdir('data')
        res_combined_fin.to_csv('data/patient_info_fin.csv')
        risk_visit.to_csv('data/Survival_Data.csv')
        total_combined.to_csv('data/all_visits.csv')

    if(plot):
        plt.figure()
    drop_ID_impute = []
    for ID, num_visits in collections.Counter(total_combined.index).items():
        if (num_visits < min_required_visits):
            drop_ID_impute.append(ID)
    total_combined = total_combined.drop(index=drop_ID_impute)

    count_values = np.array([list(collections.Counter(total_combined[total_combined['EVENT'] == 0].index).values()),
                             list(collections.Counter(total_combined[total_combined['EVENT'] == 1].index).values())])

    if(plot):
        plt.hist(count_values, bins=range(1, 16), alpha=0.7, stacked=True)
        plt.xlabel('Number of Visits')
        plt.ylabel('Number of Patients')
        plt.title('Number of Visits per Patient')
        plt.legend(['Censored', 'Uncensored'])
        plt.xlim([0, 16])
        plt.show()

    return total_combined, risk_visit, col_names

def clean_total_combined(total_combined, risk_visit, col_names, min_features_per_visit = 25):
    binary_categories = ['SEX', 'DIAB', 'CANCER', 'ANGINA', 'drusen_re', 'drusen_le', 'any_ga_re', 'any_ga_le',
                         'cga_re', 'cga_le', 'depig_re', 'depig_le', 'inpig_re', 'inpig_le', 'pig_re', 'pig_le']
    binary_category_index = []

    for category in binary_categories:
        binary_category_index.append(
            list(np.where([category in i for i in col_names])[0]))  # grepl to find which columns have DIAB in it
    binary_category_index = [item for sublist in binary_category_index for item in sublist]  # flatten list
    binary_categories_list = col_names[binary_category_index]

    non_binary_index = []
    for i in range(3, total_combined.shape[1]):
        if (i not in binary_category_index):
            non_binary_index.append(i)

    non_binary_categories_list = col_names[non_binary_index]
    scaled_features = col_names[3:]

    for i in binary_categories_list:
        if 'N' in total_combined[i].unique():
            total_combined[i] = total_combined[i].replace(['N'], 0)
            total_combined[i] = total_combined[i].replace(['Y'], 1)
        if 'F' in total_combined[i].unique():
            total_combined[i] = total_combined[i].replace(['M'], 0)
            total_combined[i] = total_combined[i].replace(['F'], 1)

    frame_order = ['VISNO_EVENT', 'EVENT', 'VISNO', 'BMI', 'SMK', 'SEX', 'DIAB', 'CANCER', 'ANGINA', 'REDRSZWI',
                   'LEDRSZWI', 'REGEOAWI', 'LEGEOAWI', 'REGEOACT', 'LEGEOACT', 'REGEOACS', 'LEGEOACS', 'RESUBFF2',
                   'LESUBFF2', 'RENDRUF2', 'LENDRUF2', 'RESSRF2', 'LESSRF2', 'RESUBHF2', 'LESUBHF2', 'RERPEDWI',
                   'LERPEDWI', 'REINCPWI', 'LEINCPWI', 'AMDSEVLE', 'AMDSEVRE', 'SCALE', 'drusen_re', 'drusen_le',
                   'any_ga_re', 'any_ga_le', 'cga_re', 'cga_le', 'depig_re', 'depig_le', 'inpig_re', 'inpig_le',
                   'pig_re', 'pig_le', 'AMDSTAT', 'ENROLLAGE', 'RACE1', 'RACE2', 'RACE3', 'RACE4', 'RACE5', 'SCHOOL1',
                   'SCHOOL2', 'SCHOOL3', 'SCHOOL4', 'SCHOOL5', 'SCHOOL6']
    visits = total_combined['VISNO'].unique()[::-1]

    if (os.path.isdir('data/1_not_scaled_data') == False):
        os.mkdir('data/1_not_scaled_data')
    if (os.path.isdir('data/2_cleaned_data') == False):
        os.mkdir('data/2_cleaned_data')

    print('1_not_scaled_data and 2_cleaned_data')
    for visit in visits:
        # initialize
        visno = visit
        export_df = pandas.DataFrame(index=total_combined.loc[total_combined['VISNO'] == '00'].index,
                                     columns=frame_order)
        current_visit = total_combined.loc[total_combined['VISNO'] == visno]
        col_names = total_combined.columns
        scaled_features = col_names[3:]
        export_df[['VISNO_EVENT', "EVENT"]] = risk_visit

        # drop patients with <xx features at this visit
        num_features_per_patient = current_visit.T.count()
        drop_ID_per_visit = num_features_per_patient[
            num_features_per_patient < min_features_per_visit].index  # drops patients who have fewer than 15 features in this visit
        current_visit = current_visit.drop(index=drop_ID_per_visit)

        # Dummy Race and School
        race = pandas.get_dummies(current_visit['RACE'])  # turns categorical RACE into sub binary categories
        race.columns = ['RACE' + str(num) for num in race.keys().values]
        current_visit = pandas.concat([current_visit, race], axis=1, join='inner')
        scaled_features = scaled_features.union(race.columns)

        school = pandas.get_dummies(current_visit['SCHOOL'])
        school.columns = ['SCHOOL' + str(num) for num in school.keys().values]
        current_visit = pandas.concat([current_visit, school], axis=1, join='inner')
        scaled_features = scaled_features.union(school.columns)

        current_visit.to_csv(str('data/1_not_scaled_data/baseline_' + visno + '_not_scaled.csv'))

        # scaler = StandardScaler().fit(current_visit[scaled_features])
        # current_visit[scaled_features] = scaler.transform(current_visit[scaled_features])

        ###
        col_names = current_visit.columns
        for col in col_names:
            if (any(pandas.notnull(current_visit[col])) == False):
                # print(col +" DROPPED BECAUSE EMPTY COLUMN "+ visit)
                current_visit = current_visit.drop(columns=col)  # Drop empty columns

        current_visit = current_visit.drop(columns='RACE')
        current_visit = current_visit.drop(columns='SCHOOL')
        current_visit = current_visit.drop(columns='BMI_R')
        current_visit = current_visit.drop(columns='CANCERTP', errors='ignore')

        # Replace BMI00, BMI01, BMI02... to BMI
        visit_categories = ['BMI', 'SMK', 'DIAB', 'CANCER', 'ANGINA']
        for category in visit_categories:
            cat_name = current_visit.columns[
                [category in i for i in current_visit.columns]]  # all categories that have BMI in it
            if (cat_name.empty == False):  # if that category has data,
                current_visit = current_visit.rename(columns={
                    str(cat_name[0]): str(category)})  # rename that category to simple BMI or SMK, removing numbers

        for col in current_visit.columns[2:]:
            if (col in export_df.columns):
                export_df[col] = current_visit[col]

        export_df = export_df.drop(columns='VISNO')
        # export_df = export_df.drop(columns='SCALE')


        export_df.to_csv(str('data/2_cleaned_data/baseline_' + visno + '.csv'))

    return frame_order, binary_categories, non_binary_categories_list

def Impute_data(total_combined, frame_order, binary_categories, non_binary_categories_list, min_features_per_visit = 25):
    # Initialize Impute Data
    visits = total_combined['VISNO'].unique()[::-1]
    warnings.filterwarnings(action='ignore')
    total_combined_imputed = pandas.DataFrame(columns=frame_order)  # index = total_combined.index,

    for visno in total_combined['VISNO'].unique():
        visit_data = pandas.read_csv(str('data/2_cleaned_data/baseline_' + visno + '.csv'), index_col=0)
        valid_visit_data = visit_data[visit_data.T.count() > min_features_per_visit]
        valid_visit_data['VISNO'] = visno
        total_combined_imputed = total_combined_imputed.append(valid_visit_data)
    total_combined_imputed['Subj ID'] = total_combined_imputed.index
    total_combined_imputed = total_combined_imputed.sort_values(['Subj ID', 'VISNO'])
    total_combined_imputed = total_combined_imputed.drop(columns='Subj ID')

    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return pandas.isnull(y), lambda z: z.nonzero()[0]

    Average_per_visit = True
    binary_categories.append('SMK')
    non_binary_categories = [cat for cat in frame_order if cat not in binary_categories]

    subject_IDs = total_combined_imputed.index.unique()
    # np.sum(total_combined_imputed.loc[subject_IDs[0]].isna(), axis=0)
    total_missing_per_feature = total_combined_imputed.isnull().sum()
    total_missing_per_visit = total_combined_imputed.isnull().sum(axis=1)
    unable_to_impute = []

    for ID in subject_IDs:
        categories_to_impute = total_combined_imputed.columns[total_combined_imputed.loc[ID].isnull().sum() > 0]
        for category in categories_to_impute:
            if (category in non_binary_categories):
                y = np.array(total_combined_imputed.loc[ID][category])
                nans, x = nan_helper(y)
                if (all(pandas.isnull(y))):  # all empty
                    # print(category + "for patient " + str(ID))
                    unable_to_impute.append([ID, category])
                elif (np.sum(nans) == 1):  # 1 empty
                    if (np.where(nans)[0] == 2):
                        y[2] = (y[1] - y[0]) / 2 + y[1]
                    elif (np.where(nans)[0] == 1):
                        y[1] = 2 * (y[2] - y[0]) / 3 + y[0]
                    else:
                        y[0] = y[1] - (y[2] - y[1]) * 2
                    total_combined_imputed.loc[ID][category] = y
                else:  # 2 empty
                    y_new = np.copy(y)
                    visit_nums = np.array([int(i) for i in list(total_combined_imputed.loc[ID]['VISNO'])])
                    non_nan_count = 0
                    for i in range(len(y)):
                        if (i in np.where(~nans)[0]):
                            non_nan_count += 1
                        else:
                            if (i < max(np.where(~nans)[0])):
                                y_new[i] = y[~nans][non_nan_count - 1] + (
                                            y[~nans][non_nan_count] - y[~nans][non_nan_count - 1]) * (
                                                       visit_nums[i] - visit_nums[~nans][non_nan_count - 1]) / (
                                                       visit_nums[~nans][non_nan_count] - visit_nums[~nans][
                                                   non_nan_count - 1])
                            else:
                                y_new[i] = y[~nans][non_nan_count - 1]
                    total_combined_imputed.loc[ID][category] = y_new
            #                 if(pandas.isnull(y).sum()>0):
            #                     print(category + "for patient " + str(ID))
            else:
                unfilled_visits = []
                filled_visits = []
                unfilled_visits = total_combined_imputed.loc[ID]['VISNO'].iloc[
                    np.where(total_combined_imputed.loc[ID][category].isnull())[0]].astype(int).values
                filled_visits = total_combined_imputed.loc[ID]['VISNO'].iloc[
                    np.where(total_combined_imputed.loc[ID][category].isnull() == False)[0]].astype(int).values
                # print(len(filled_visits))
                if (len(filled_visits)):
                    for i in unfilled_visits:
                        closest_visit = filled_visits[np.argmin(abs(filled_visits - i))]
                        fill_value = total_combined_imputed.loc[ID][category][
                            total_combined_imputed.loc[ID]['VISNO'] == str(closest_visit).zfill(2)]
                        total_combined_imputed.loc[ID][category][
                            total_combined_imputed.loc[ID]['VISNO'] == str(i).zfill(2)] = fill_value
                else:
                    # print(category + "for patient " + str(ID))
                    unable_to_impute.append([ID, category])

    total_not_imputed_per_feature = total_combined_imputed.isnull().sum()
    percent_imputed_feature = 1 - total_not_imputed_per_feature / total_missing_per_feature
    total_combined_imputed.to_csv('data/imputed_by_patient.csv')

    if (Average_per_visit):
        # Fill in missing data to most frequent or average of visit
        total_combined_imputed['Subj ID'] = total_combined_imputed.index
        total_combined_imputed.index = np.arange(total_combined_imputed.shape[0])

        bin_cat = binary_categories
        bin_cat.append('SMK')
        non_bin = list(non_binary_categories_list)
        non_bin.append('BMI')

        for visit in visits:
            for missing_cat in total_combined_imputed.columns[
                total_combined_imputed[total_combined_imputed['VISNO'] == visit].isnull().any()]:
                if missing_cat in bin_cat:
                    idx = total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat][
                        total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat].isnull()].index
                    value = collections.Counter(
                        total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat]).most_common(1)[
                        0][1]
                    total_combined_imputed[missing_cat].loc[idx] = value
                if missing_cat in non_bin:
                    idx = total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat][
                        total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat].isnull()].index
                    value = round(
                        np.mean(total_combined_imputed[total_combined_imputed['VISNO'] == visit][missing_cat]), 2)
                    total_combined_imputed[missing_cat].loc[idx] = value

        total_combined_imputed.index = total_combined_imputed['Subj ID']
        total_combined_imputed = total_combined_imputed.drop(columns=['Subj ID'])
        total_combined_imputed.to_csv('data/imputed_with_avg.csv')

    impute_stats = pandas.DataFrame(columns={'Number of Features Not Imputed', 'Percent of Empty Features Imputed'})
    impute_stats['Number of Features Not Imputed'] = total_not_imputed_per_feature
    impute_stats['Percent of Empty Features Imputed'] = percent_imputed_feature
    print(impute_stats)
    print('%0.3f percent of the data is censored' % (
                (1 - np.mean(total_combined_imputed[total_combined_imputed['VISNO'] == '00']['EVENT'])) * 100))

    a = total_not_imputed_per_feature / 3
    a.to_csv('data/number_of_missing_this_category.csv')
    a

    imputed_no_avg = pandas.read_csv('data/imputed_by_patient.csv', index_col=0)
    imputed_yes_avg = pandas.read_csv('data/imputed_with_avg.csv', index_col=0)
    imputed_no_avg['VISNO'] = imputed_no_avg['VISNO'].map("{0:0=2d}".format)
    imputed_yes_avg['VISNO'] = imputed_yes_avg['VISNO'].map("{0:0=2d}".format)
    imputed_no_avg = imputed_no_avg[imputed_no_avg['VISNO_EVENT'].astype(int) > 6]
    imputed_yes_avg = imputed_yes_avg[imputed_yes_avg['VISNO_EVENT'].astype(int) > 6]

    drop_bc_not_complete = imputed_no_avg[pandas.isnull(imputed_no_avg).sum(axis=1) > 0].index.unique()
    imputed_no_avg = imputed_no_avg.drop(index=drop_bc_not_complete)

    print('3_imputed_no_avg and 4_imputed_yes_avg')
    visits = imputed_no_avg['VISNO'].unique()
    if (os.path.isdir('data/3_imputed_no_avg') == False):
        os.mkdir('data/3_imputed_no_avg')
    if (os.path.isdir('data/4_imputed_yes_avg') == False):
        os.mkdir('data/4_imputed_yes_avg')
    for visit in visits:
        current_no_avg = imputed_no_avg[imputed_no_avg['VISNO'] == visit]
        current_yes_avg = imputed_yes_avg[imputed_yes_avg['VISNO'] == visit]
        current_no_avg.to_csv('data/3_imputed_no_avg/not_scaled_' + visit + '.csv')
        current_yes_avg.to_csv('data/4_imputed_yes_avg/not_scaled_' + visit + '.csv')

    col_names = imputed_no_avg.columns
    scaled_features = col_names[3:]
    scaler_no_avg = StandardScaler().fit(imputed_no_avg[scaled_features])
    imputed_no_avg[scaled_features] = scaler_no_avg.transform(imputed_no_avg[scaled_features])

    scaler_yes_avg = StandardScaler().fit(imputed_yes_avg[scaled_features])
    imputed_yes_avg[scaled_features] = scaler_yes_avg.transform(imputed_yes_avg[scaled_features])

    print('5_imputed_no_avg_scaled and 6_imputed_yes_avg_scaled')
    if (os.path.isdir('data/5_imputed_no_avg_scaled') == False):
        os.mkdir('data/5_imputed_no_avg_scaled')
    if (os.path.isdir('data/6_imputed_yes_avg_scaled') == False):
        os.mkdir('data/6_imputed_yes_avg_scaled')
    for visit in visits:
        current_no_avg = imputed_no_avg[imputed_no_avg['VISNO'] == visit]
        current_yes_avg = imputed_yes_avg[imputed_yes_avg['VISNO'] == visit]
        current_no_avg.to_csv('data/5_imputed_no_avg_scaled/scaled_' + visit + '.csv')
        current_yes_avg.to_csv('data/6_imputed_yes_avg_scaled/scaled_' + visit + '.csv')

def plot_figures(total_combined):
    positive_patients = total_combined[total_combined['EVENT'] == 1].index.unique()
    negative_patients = total_combined[total_combined['EVENT'] == 0].index.unique()

    temp.loc[positive_patients]

    start_at_0 = temp[temp['00'] == 0].index
    start_at_1 = temp[temp['00'] == 1].index
    start_at_2 = temp[temp['00'] == 2].index
    start_at_3 = temp[temp['00'] == 3].index
    start_at_4 = temp[temp['00'] == 4].index

    starting_indicies = [start_at_0, start_at_1, start_at_2, start_at_3, start_at_4]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in np.arange(len(starting_indicies)):
        start_temp = temp.loc[starting_indicies[i]]
        mean_scale = np.mean(start_temp, axis=0)
        sd_scale = np.std(start_temp, axis=0)
        x = mean_scale.index
        x = [int(j) for j in x]
        axs[i // 3, i % 3].errorbar(x, mean_scale, sd_scale, linestyle='None', fmt='o', ecolor='black', elinewidth=1,
                                    capsize=3, capthick=1)
        axs[i // 3, i % 3].set(xlabel='Visit Number')
        axs[i // 3, i % 3].set(ylabel='Scale')
        axs[i // 3, i % 3].set_ylim(0, 5.5)
        axs[i // 3, i % 3].set_title('Patients starting at scale %d, n = %d' % (i, len(starting_indicies[i])))

    # of patients who reach 5, plot who starts where and how they progress

    five_idx = risk_visit[risk_visit['EVENT'] == 1].index
    Five_0_idx = [idx for idx in starting_indicies[0] if idx in five_idx]
    Five_1_idx = [idx for idx in starting_indicies[1] if idx in five_idx]
    Five_2_idx = [idx for idx in starting_indicies[2] if idx in five_idx]
    Five_3_idx = [idx for idx in starting_indicies[3] if idx in five_idx]
    Five_4_idx = [idx for idx in starting_indicies[4] if idx in five_idx]

    reach_5_idx = [Five_0_idx, Five_1_idx, Five_2_idx, Five_3_idx, Five_4_idx]
    x_y_time = pandas.DataFrame(index=total_combined['VISNO'].unique())
    for i, idx in enumerate(reach_5_idx):
        x = total_combined.loc[idx]['VISNO']
        y = total_combined.loc[idx]['SCALE']
        x_y_temp = []

        for visno in x.unique():
            mean_val = np.mean(y[x == visno])
            x_y_temp.append([int(visno), mean_val])

        x_y_temp = pandas.DataFrame(x_y_temp)
        x_y_temp.columns = ['visno', str('scale' + str(i))]
        x_y_temp.index = x_y_temp['visno']
        x_y_temp = x_y_temp.drop(columns='visno')

        x_y_time = pandas.concat([x_y_time, x_y_temp], axis=1)

    # plt.plot(x_y_time['visno'], x_y_time['scale'])
    # x_y_time[1:3][0]

    for idx in x_y_time.index:
        if (all(x_y_time.loc[idx].isnull())):
            x_y_time = x_y_time.drop(index=idx)

    for col in x_y_time.columns:
        values = x_y_time[col][np.isfinite(x_y_time[col])]
        axs[1, 2].plot(values)

    scales = [str(val + ' n=') for val in x_y_time.columns.values]
    nums = [str(len(length)) for length in reach_5_idx]
    leg = np.char.add(scales, nums)
    axs[1, 2].legend(leg)
    axs[1, 2].set_title('Development of patients that reach 5')
    plt.xlabel('Visit Number')
    plt.show()

    # plt.savefig('plots/errorbar_of_scale_depending_on_visno_00_scale.pdf')
#####MAIN

def clean_json(plot=False):
    warnings.filterwarnings(action='ignore')

    data = load_json('AREDS_participants_amd3.json')
    total_combined, risk_visit, col_names = rearrange_json(data, plot=plot)
    frame_order, binary_categories, non_binary_categories_list = clean_total_combined(total_combined, risk_visit, col_names)
    Impute_data(total_combined, frame_order, binary_categories, non_binary_categories_list)
    if(plot):
        plot_figures(total_combined)