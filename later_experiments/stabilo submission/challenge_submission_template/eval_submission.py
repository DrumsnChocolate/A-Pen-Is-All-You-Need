import requests
import pandas as pd
from Levenshtein import distance as levenshtein_distance

import os


def split_equations(sensor_df: pd.DataFrame, label_df: pd.DataFrame) -> list:
    '''
    Splits the given sensor data according to the given label data
    :param sensor_df: a pandas dataframe containing the sensor data with their timestamps
    :param label_df: a pandas dataframe containing the labels and their timestamps
    :returns: a list containing tuples of (label: string, sensor_data: list of sensor sample dict)
    '''
    labels_with_data = []
    for _, elem_row in label_df.iterrows():
        label = elem_row[0]
        start = int(elem_row[1])
        stop = int(elem_row[2])
        sensor_data_df = sensor_df[(sensor_df['Millis'] >= start) & (sensor_df['Millis'] < stop)]
        # convert to list of sensorsample dicts
        samples = []
        for _, line in sensor_data_df.iterrows():
            samples.append({
                'a1x': line[1],
                'a1y': line[2],
                'a1z': line[3],
                'a2x': line[4],
                'a2y': line[5],
                'a2z': line[6],
                'gx': line[7],
                'gy': line[8],
                'gz': line[9],
                'mx': line[10],
                'my': line[11],
                'mz': line[12],
                'force': line[13],
                'counter': line[14],
                'timestamp': line[0]
            })

        labels_with_data.append((label, samples))
    return labels_with_data


def read_dataset(path_to_data_folder: str, stop_after_one_person: bool = False, stop_after_n=1) -> list:
    '''
    Gets the super folder path and reads all (recording) subfolders for each (person) subfolder, gathering the sensor and label data.
    Returns a list containing one element per person. Each element (equation) contains all labels and the according raw sensor data.
    '''
    person_folders = [f.path for f in os.scandir(path_to_data_folder) if f.is_dir()]
    all_persons_data = []  # will contain one element per person. each element will contain all labels and the according sensor data of one person
    for person_num, person_folder in enumerate(person_folders):
        print(f'Collecting data for person {person_num}...')
        recording_folders = [f.path for f in os.scandir(person_folder) if f.is_dir()]
        person_data = []  # will contain all labels and the according sensor data of one person
        for rec_num, recording_folder in enumerate(recording_folders):
            # read sensor and label data
            label_df = pd.read_csv(recording_folder + '/labels.csv', encoding='utf-8', header=0, delimiter=';')
            sensor_df = pd.read_csv(recording_folder + '/sensor_data.csv', encoding='utf-8', header=0, delimiter=';')

            # split sensor data according to label timestamps
            split_labels_with_data = split_equations(sensor_df, label_df)

            print(f'-- Recording {rec_num}: {len(split_labels_with_data)} equations')
            person_data.extend(split_labels_with_data)

        all_persons_data.append(person_data)

        if stop_after_one_person or person_num == stop_after_n - 1:
            break

    print('\nSummary:')
    print(f'Persons: {len(all_persons_data)}')
    print(f'Overall equations: {sum([len(person_data) for person_data in all_persons_data])}')
    return all_persons_data


def eval_submission(host: str, path_to_data_folder: str, stop_after_one_person: bool = False, stop_after_n=1) -> None:
    # read all files first. Very similar to read_dataset.py
    all_persons_data = read_dataset(path_to_data_folder, stop_after_one_person, stop_after_n)

    # prepare REST API requests
    correct_equation = 0
    wrong_equation = 0
    lev_distances = []
    for person_i, person in enumerate(all_persons_data):
        # the first five equations are the labelled adaptation equations
        adaptation_set = [{'label': tuple[0], 'data': tuple[1]} for tuple in person[0:5]]

        # the rest are unlabelled validation equations
        validation_set = [{'data': tuple[1]} for tuple in person[5:]]
        ground_truths = [tuple[0] for tuple in person[5:]]

        # send out request for this person
        request = requests.post(
            host + 'ubicomp_challenge_21/predict',
            json={'adaptation': adaptation_set, 'validation': validation_set}
        )

        print("person nr.", person_i, "- status code:", request.status_code)
        hypos = request.json()['hypotheses']

        # compare hypotheses with ground truths
        if len(ground_truths) != len(hypos):
            print(f"WARNING: GT length: {len(ground_truths)}, hypos length: {len(hypos)}")
        for ground_truth, hypo in zip(ground_truths, hypos):
            if ground_truth == hypo:
                correct_equation += 1
            else:
                wrong_equation += 1

            lev_dist = levenshtein_distance(hypo, ground_truth)
            lev_distances.append(lev_dist)
            print('Hypo: {}, Label: {}, Lev-Distance:{}'.format(hypo, ground_truth, lev_dist))

    print("\nRESULTS:")
    avg_ld = sum(lev_distances) / len(lev_distances)
    print('Overall Levenshtein Distance: {}'.format(avg_ld))  # THIS COUNTS FOR THE CHALLENGE
    print('Overall Word accuracy: {}'.format(
        correct_equation / (wrong_equation + correct_equation)))  # this is just for fun


if __name__ == "__main__":
    host = 'http://localhost:8080/'
    folder = 'flask_app/app/STABILO_UBICOMP_CHALLENGE_2021_part1+2'  # TODO adapt this
    # the validation set folder will have exactly the same format as the training data folder

    stop_after_one_person = False  # set this to False if you want to evaluate with ALL training data
    eval_submission(host, folder, stop_after_one_person, stop_after_n=10)
