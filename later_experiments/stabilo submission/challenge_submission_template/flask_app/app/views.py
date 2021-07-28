from app import app
from flask import request

import json
import pandas

import app.transformer_solution as solution


def _list_of_sensorsample_dicts_to_df(data:list) -> pandas.DataFrame:
    '''
    Converts a given list of sensor sample dictionaries to a pandas dataframe. Info on the sensors: https://stabilodigital.com/sensors-2021/
    '''
    samples = []
    for sample in data:
        samples.append([
            float(sample['timestamp']),
            float(sample['a1x']),
            float(sample['a1y']),
            float(sample['a1z']),
            float(sample['a2x']),
            float(sample['a2y']),
            float(sample['a2z']),
            float(sample['gx']),
            float(sample['gy']),
            float(sample['gz']),
            float(sample['mx']),
            float(sample['my']),
            float(sample['mz']),
            float(sample['force']),
            float(sample['counter'])
    ])
    df = pandas.DataFrame(samples, columns=["Millis","Acc1 X","Acc1 Y","Acc1 Z","Acc2 X","Acc2 Y","Acc2 Z","Gyro X","Gyro Y","Gyro Z","Mag X","Mag Y","Mag Z","Force","Time"])
    return df

@app.route('/ubicomp_challenge_21/predict', methods=['POST'])
def predict():
    if request.json.get('adaptation') == None:
        return "no adaptation data given", 400
    adaptations_json = request.json.get('adaptation')

    if request.json.get('validation') == None:
        return "no validation data given", 400
    validations_json = request.json.get('validation')

    # prepare adaptation set
    adaptation_df_and_labels = [] # will contain tuples of label, data
    for adaptation_eq in adaptations_json:
        label = adaptation_eq['label']
        data = adaptation_eq['data'] # list of sensor sample dicts
        df = _list_of_sensorsample_dicts_to_df(data)
        adaptation_df_and_labels.append((label, df))

    # prepare validation set
    validation_dfs = []
    for validation_eq in validations_json:
        data = validation_eq['data'] # list of sensor sample dicts
        df = _list_of_sensorsample_dicts_to_df(data)
        validation_dfs.append(df)

    # print some example data
    print("adaptation equation 0:")
    print(adaptation_df_and_labels[0][0])
    print(adaptation_df_and_labels[0][1].head())

    print("valdidation equation 0:")
    print(validation_dfs[0].head())

    ###################
    # DO MAGIC by adapting pipeline or model
    # and add your requirements to requirements.txt
    ###################

    # prediction
    hypotheses = []
    for validation_eq in validation_dfs:
        ################
        # DO MAGIC by predicting an equation
        # and add your requirements to requirements.txt
        ################
        prediction = solution.preprocess_and_predict(validation_eq).numpy().decode('utf-8')
        print(prediction)
        hypotheses.append(prediction)

    return json.dumps({'hypotheses':hypotheses}), 201
