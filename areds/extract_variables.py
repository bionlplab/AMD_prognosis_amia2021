import collections
import json
from pathlib import Path
import pandas as pd

# https://www.ncbi.nlm.nih.gov/gap/advanced_search/?OBJ=variable&COND=%7B%22study_accession%22:%5B%22phs000001.v3.p1%22%5D%7D
from areds.amd_calculator import drusen, any_ga, cga, depigmentation, increased_pigment, pigment


def get_helper_baseline(df, participants, value_column, key_column='dbGaP SubjID'):
    cnt = collections.Counter()
    for subj, value in zip(df[key_column], df[value_column]):
        if subj not in participants:
            participants[subj] = {'dbGaP SubjID': subj, 'VISITS': {}}
            cnt['new participants'] += 1
        participants[subj][value_column] = value
    if len(cnt) > 0:
        print('New participants:', cnt['new participants'])


def year_to_visno(year: int) -> str:
    if year == 0:
        return '00'
    elif year == 1:
        return '02'
    elif year == 2:
        return '04'
    elif year == 3:
        return '06'
    elif year == 4:
        return '08'
    elif year == 5:
        return '10'
    elif year == 6:
        return '12'
    elif year == 7:
        return '14'
    elif year == 8:
        return '16'
    elif year == 9:
        return '18'
    elif year == 10:
        return '20'
    elif year == 11:
        return '22'
    elif year == 12:
        return '24'
    elif year == 13:
        return '26'
    else:
        raise KeyError


def get_helper_year(df, participants, year, value_column, key_column='dbGaP SubjID'):
    cnt = collections.Counter()
    visno = year_to_visno(year)
    for subj, value in zip(df[key_column], df[value_column]):
        if subj not in participants:
            participants[subj] = {'dbGaP SubjID': subj, 'VISITS': {}}
            cnt['new participants'] += 1
        participant = participants[subj]
        if visno not in participant['VISITS']:
            participant['VISITS'][visno] = {'VISNO': visno}
            cnt['new visnos'] += 1
        visit = participant['VISITS'][visno]
        visit[value_column] = value
    # if len(cnt) > 0:
    #     print('New participants:', cnt['new participants'])
    #     print('New visnos:', cnt['new visnos'])


def get_helper_fundus(df, participants, value_column, key_column='dbGaP SubjID'):
    cnt = collections.Counter()
    for subj, visno, value in zip(df[key_column], df['VISNO'], df[value_column]):
        if subj not in participants:
            participants[subj] = {}
            cnt['new participants'] += 1
        participant = participants[subj]
        if visno not in participant['VISITS']:
            participant['VISITS'][visno] = {'VISNO': visno}
            cnt['new visnos'] += 1
        visit = participant['VISITS'][visno]
        visit[value_column] = value
    if len(cnt) > 0:
        print('New participants:', cnt['new participants'])
        print('New visnos:', cnt['new visnos'])


def get_followup(dir, participants):
    genspecphenotype_gru_df = pd.read_csv(
        dir / 'phs000001.v3.pht000001.v2.p1.c2.genspecphenotype.GRU.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    genspecphenotype_edo_df = pd.read_csv(
        dir / 'phs000001.v3.pht000001.v2.p1.c1.genspecphenotype.EDO.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    genspecphenotype_df = pd.concat([genspecphenotype_gru_df, genspecphenotype_edo_df], axis=0)

    # BODY-MASS INDEX AT YEAR 3 (PARTICIPANTS WITH A GENETIC SPECIMEN)
    for year in [0, 3, 4, 5, 6, 7, 8, 9, 10]:
        get_helper_year(genspecphenotype_df, participants, year, value_column='BMI%02d' % year)

    # SMOKING STATUS AT YEAR 3 - BASED ON HAVING SMOKED FOR 6 MONTHS (PARTICIPANTS WITH A GENETIC SPECIMEN)
    # 1 - never, 2 - former, 3 - current
    for year in [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        get_helper_year(genspecphenotype_df, participants, year, value_column='SMK%02d' % year)

    # CURRENTLY HAVE DIABETES AT YEAR 1 (PARTICIPANTS WITH A GENETIC SPECIMEN)
    for year in range(0, 9):
        get_helper_year(genspecphenotype_df, participants, year, value_column='DIAB%02d' % year)

    # NEW DIAGNOSIS OF CANCER (SINCE LAST STUDY VISIT) AT YEAR 1 (PARTICIPANTS WITH A GENETIC SPECIMEN)
    for year in range(0, 7):
        get_helper_year(genspecphenotype_df, participants, year, value_column='CANCER%02d' % year)

    # HISTORY OF ANGINA AT YEAR 3 (PARTICIPANTS WITH A GENETIC SPECIMEN)
    for year in [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        get_helper_year(genspecphenotype_df, participants, year, value_column='ANGINA%02d' % year)


def get_demographic(dir, participants):
    enrollment_gru_df = pd.read_csv(
        dir / 'phs000001.v3.pht000373.v2.p1.c2.enrollment_randomization.GRU.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    enrollment_edo_df = pd.read_csv(
        dir / 'phs000001.v3.pht000373.v2.p1.c1.enrollment_randomization.EDO.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    enrollment_df = pd.concat([enrollment_gru_df, enrollment_edo_df], axis=0)

    genspecphenotype_gru_df = pd.read_csv(
        dir / 'phs000001.v3.pht000001.v2.p1.c2.genspecphenotype.GRU.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    genspecphenotype_edo_df = pd.read_csv(
        dir / 'phs000001.v3.pht000001.v2.p1.c1.genspecphenotype.EDO.txt',
        comment='#', sep='\t', dtype={'dbGaP SubjID': str})
    genspecphenotype_df = pd.concat([genspecphenotype_gru_df, genspecphenotype_edo_df], axis=0)

    # AGE AT BASELINE (ALL PARTICIPANTS)
    get_helper_baseline(enrollment_df, participants, value_column='ID2')

    # AGE AT BASELINE (ALL PARTICIPANTS)
    get_helper_baseline(enrollment_df, participants, value_column='ENROLLAGE')

    # SEX (ALL PARTICIPANTS)
    get_helper_baseline(enrollment_df, participants, value_column='SEX')

    # HIGHEST LEVEL OF SCHOOL ATTENDED (ALL PARTICIPANTS)
    # 1 - grade 11 or less, 2 - high school, 3 - college, -4 bachelor, 5 - postgraduate, 6 - refused to answer
    get_helper_baseline(enrollment_df, participants, value_column='SCHOOL')

    # RACE OR ETHNIC BACKGROUND (ALL PARTICIPANTS)
    # 1 - white, 2 - black, 3 - hispanic, 4 - asian, 5 - other
    get_helper_baseline(enrollment_df, participants, value_column='RACE')

    # BODY-MASS INDEX AT BASELINE (ALL PARTICIPANTS)
    get_helper_baseline(enrollment_df, participants, value_column='BMI_R')
    get_helper_baseline(genspecphenotype_df, participants, value_column='BMI00')

    # SMOKING STATUS AT BASELINE - BASED ON HAVING SMOKED FOR 6 MONTHS (PARTICIPANTS WITH A GENETIC SPECIMEN)
    # 1 - never, 2 - former, 3 - current
    get_helper_baseline(genspecphenotype_df, participants, value_column='SMK00')

    # HISTORY OF DIABETES AT BASELINE (PARTICIPANTS WITH A GENETIC SPECIMEN)
    get_helper_baseline(genspecphenotype_df, participants, value_column='DIAB00')

    # TYPE OF CANCER (ALL PARTICIPANTS)
    # 1 - breast, 2 - colon, 3 - lung, 4 - prostate, 5 - melanoma, 6 - basal, 7 - other, 8 - multiple types
    get_helper_baseline(enrollment_df, participants, value_column='CANCERTP')
    #  HISTORY OF CANCER AT BASELINE (PARTICIPANTS WITH A GENETIC SPECIMEN)
    get_helper_baseline(genspecphenotype_df, participants, value_column='CANCER00')

    # HISTORY OF ANGINA AT BASELINE (PARTICIPANTS WITH A GENETIC SPECIMEN)
    get_helper_baseline(genspecphenotype_df, participants, value_column='ANGINA00')


# https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/dataset.cgi?study_id=phs000001.v3.p1&pht=375
def get_fundus(dir, participants):
    fundus_df = pd.read_csv(dir / 'phs000001.v3.pht000375.v2.p1.c2.fundus.GRU.txt',
                            comment='#', sep='\t', dtype={'dbGaP SubjID': str, 'VISNO': str})

    #  MAXIMUM DRUSEN SIZE W/I GRID RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='REDRSZWI')
    #  MAXIMUM DRUSEN SIZE W/I GRID LE EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LEDRSZWI')
    # GEOGRAPHIC ATROPHY W/I GRID RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='REGEOAWI')
    # GEOGRAPHIC ATROPHY W/I GRID LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LEGEOAWI')
    # GEOGRAPHIC ATROPHY CENTER POINT RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='REGEOACT')
    # GEOGRAPHIC ATROPHY CENTER POINT LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LEGEOACT')
    # GEOGRAPHIC ATROPHY AREA C/SUB RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='REGEOACS')
    # GEOGRAPHIC ATROPHY AREA C/SUB LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LEGEOACS')
    # SUBRETINAL FIBROSIS FIELD 2 RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='RESUBFF2')
    # SUBRETINAL FIBROSIS FIELD 2 LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LESUBFF2')
    # NON-DRUSENOID PED FIELD 2 RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='RENDRUF2')
    # NON-DRUSENOID PED FIELD 2 LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LENDRUF2')
    # SSR/HEMORRHAGIC RD FIELD 2 RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='RESSRF2')
    # SSR/HEMORRHAGIC RD FIELD 2 LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LESSRF2')
    # SUBRETINAL/SUBRPE HEMORRHAGE FIELD 2 RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='RESUBHF2')
    # SUBRETINAL/SUBRPE HEMORRHAGE FIELD 2 LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LESUBHF2')
    # RPE DEPIGMENTATION AREA W/I GRID RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='RERPEDWI')
    # RPE DEPIGMENTATION AREA W/I GRID LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LERPEDWI')
    # INCREASED PIGMENT AREA W/I GRID RT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='REINCPWI')
    # INCREASED PIGMENT AREA W/I GRID LT EYE (ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='LEINCPWI')

    # AMD SEVERITY SCALE SCORE LT EYE, 13 steps
    get_helper_fundus(fundus_df, participants, value_column='AMDSEVLE')
    # AMD SEVERITY SCALE SCORE RT EYE, 13 steps
    get_helper_fundus(fundus_df, participants, value_column='AMDSEVRE')
    # AMD SIMPLE SCALE SCORE (PERSON; ALL PARTICIPANTS)
    get_helper_fundus(fundus_df, participants, value_column='SCALE')


def get_amd(participants):
    for participant in participants.values():
        for visit in participant['VISITS'].values():
            try:
                visit['drusen_re'] = drusen(visit['REDRSZWI'])
                visit['drusen_le'] = drusen(visit['LEDRSZWI'])
                visit['any_ga_re'] = any_ga(visit['REGEOAWI'])
                visit['any_ga_le'] = any_ga(visit['LEGEOAWI'])
                visit['cga_re'] = cga(visit['REGEOACT'], visit['REGEOACS'], visit['RESUBFF2'])
                visit['cga_le'] = cga(visit['LEGEOACT'], visit['LEGEOACS'], visit['LESUBFF2'])
                visit['depig_re'] = depigmentation(visit['RERPEDWI'])
                visit['depig_le'] = depigmentation(visit['LERPEDWI'])
                visit['inpig_re'] = increased_pigment(visit['REINCPWI'])
                visit['inpig_le'] = increased_pigment(visit['LEINCPWI'])
                visit['pig_re'] = pigment(visit['depig_re'], visit['inpig_re'], visit['cga_re'], visit['any_ga_re'])
                visit['pig_le'] = pigment(visit['depig_le'], visit['inpig_le'], visit['cga_le'], visit['any_ga_le'])
            except:
                continue


def extract_variables():
    top_dir = Path.home() / 'data/dbGaP-25378-phenotype'
    participants = {}
    get_demographic(top_dir, participants)
    get_followup(top_dir, participants)
    get_fundus(top_dir, participants)
    get_amd(participants)

    with open(top_dir / '/AREDS_participants_amd.json', 'w') as fp:
        # objs = sorted(participants.values(), key=lambda x: x['dbGaP SubjID'])
        objs = list(participants.values())
        for obj in objs:
            visits = obj['VISITS']
            del obj['VISITS']
            obj['VISITS'] = sorted(visits.values(), key=lambda v: v['VISNO'])
        json.dump(objs, fp, indent=2)
