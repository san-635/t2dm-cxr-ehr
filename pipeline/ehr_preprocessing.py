import numpy as np
import os
import re
import pandas as pd

# GLOBAL VARIABLES #

# encoding for biological sex
gender_map = {'F': 1, 'M': 2}

# ICD-9-CM and ICD-10-CM codes that correspond to T2DM
diagnosis_labels = ['25000','25010','25020','25030','25040','25050','25060','25070','25080','25090',
                    '25002','25012','25022','25032','25042','25052','25062','25072','25082','25092',
                    'E11','E110','E1100','E1101','E111','E1110','E1111','E112','E1121','E1122','E1129',
                    'E113','E1131','E11311','E11319','E1132','E11321','E113211','E113212','E113213',
                    'E113219','E11329','E113291','E113292','E113293','E113299','E1133','E11331','E113311',
                    'E113312','E113313','E113319','E11339','E113391','E113392','E113393','E113399','E1134',
                    'E11341','E113411','E113412','E113413','E113419','E11349','E113491','E113492','E113493',
                    'E113499','E1135','E11351','E113511','E113512','E113513','E113519','E11352','E113521',
                    'E113522','E113523','E113529','E11353','E113531','E113532','E113533','E113539','E11354',
                    'E113541','E113542','E113543','E113549','E11355','E113551','E113552','E113553','E113559',
                    'E11359','E113591','E113592','E113593','E113599','E1136','E1137','E1137X1','E1137X2',
                    'E1137X3','E1137X9','E1139','E114','E1140','E1141','E1142','E1143','E1144','E1149','E115',
                    'E1151','E1152','E1159','E116','E1161','E11610','E11618','E1162','E11620','E11621','E11622',
                    'E11628','E1163','E11630','E11638','E1164','E11641','E11649','E1165','E1169','E118','E119']

# ICD-9-CM and ICD-10-CM codes that correspond to a family history of diabetes
family_history_diagnoses_labels = ['V180','Z833']

# NON-TIME SERIES EPISODIC DATA PREPROCESSING #

# encode biological sex 'F' as 1 and 'M' as 2
def transform_gender(gender_series):
    global gender_map
    return {'Gender': gender_series.apply(lambda x: gender_map[x])}

# extract diagnosis labels and family history of diabetes labels from diagnoses df
def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    global family_history_diagnoses_labels
    diagnoses['value'] = 1
    labels = diagnoses[['stay_id', 'icd_code', 'value']].drop_duplicates()\
                      .pivot(index='stay_id', columns='icd_code', values='value').fillna(0).astype(int)
    missing_cols = [l for l in diagnosis_labels if l not in labels.columns]
    missing_data = pd.DataFrame(0, index = labels.index, columns = missing_cols)
    labels = pd.concat([labels, missing_data], axis=1)
    labels = labels[diagnosis_labels]
    labels = labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)
    diagnoses['family_history_indicator'] = diagnoses['icd_code'].isin(family_history_diagnoses_labels)
    labels['Family history'] = diagnoses.groupby('stay_id')['family_history_indicator'].any().astype(int).fillna(0)
    return labels

def assemble_non_timeseries(stays, diagnoses):
    data = {'Icustay': stays.stay_id, 'Age': stays.anchor_age, 'Length of Stay': stays.los,}
    data.update(transform_gender(stays.gender))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = pd.DataFrame(data).set_index('Icustay')
    data = data[['Gender', 'Age', 'Height', 'Weight', 'Length of Stay',]]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)

# TIME SERIES EPISODIC DATA PREPROCESSING #

def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='itemid', right_on='itemid') 

def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.variable == variable)
    v = events.value[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'value'] = v
    return events.dropna(subset=['value'])

# cleaning functions
# Systolic BP (mmHg): some may be strings 'sbp/dbp' so extract first number
def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)

# Diastolic BP (mmHg): some may be strings 'sbp/dbp' so extract second number
def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)

# Temperature (C): convert Farenheit to Celsius; some Celsius are > 79 (assume Farenheit)
def clean_temperature(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'F' in s.lower()) | df.mimic_label.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v

# Weight (kg): convert pounds to kg
def clean_weight(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'lb' in s.lower()) | df.mimic_label.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v

# Height (cm): convert inches to cm
def clean_height(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'in' in s.lower()) | df.mimic_label.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v

# Heart Rate (bpm), Respiratory rate (insp/min) and Urine output (mL) do not require any cleaning

clean_fns = {
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}

def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.variable == var_name)
        try:
            events.loc[idx, 'value'] = clean_fn(events[idx])
        except Exception as e:
            print("Exception in clean_events function:", clean_fn.__name__, e)
            exit()
    return events.loc[events.value.notnull()]

def convert_events_to_timeseries(events, variable_column='variable', variables=[]):
    metadata = events[['charttime', 'stay_id']].sort_values(by=['charttime', 'stay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    timeseries = events[['charttime', variable_column, 'value']]\
                    .sort_values(by=['charttime', variable_column, 'value'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns=variable_column, values='value')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

def get_events_for_episode(timeseries, icustayid, intime=None, outtime=None):
    idx = (timeseries.stay_id == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((timeseries.charttime >= intime) & (timeseries.charttime <= outtime))
    timeseries = timeseries[idx]
    del timeseries['stay_id']
    return timeseries

def add_hours_elapsed_to_events(episode, intime, remove_charttime=True):
    episode = episode.copy()
    episode['HOURS'] = (episode.charttime - intime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del episode['charttime']
    return episode

def get_first_valid_from_timeseries(episode, variable):
    if variable in episode:
        idx = episode[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return episode[variable].iloc[loc]
    return np.nan