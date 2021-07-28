import os
import pickle
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import random

data_folder = 'app/STABILO_UBICOMP_CHALLENGE_2021_part1+2/'
preprocessed_file = 'app/preprocessed/2021'
pkl_filename = 'stabilo_challenge_2021_part1+2.pkl'
RESAMPLE = False
DESIRED_SAMPLE_LENGTH = 1330
LOG = False

MAX_ACC_FRONT = 32768
MAX_GYR = 32768
MAX_ACC_BACK = 8192
MAX_MAG = 8192
MAX_FORCE = 4096


def log(o):
    if LOG:
        print(o)


def split_elements(sensor_df, label_df):
    '''
    Splits the given sensor data according to the given label data
    :param sensor_df: a pandas dataframe containing the sensor data with their timestamps
    :param label_df: a pandas dataframe containing the labels and their timestamps
    :return: a list containing tuples of (label: string, sensor_data: pandas.DataFrame)
    '''
    labels_with_data = []
    for _, elem_row in label_df.iterrows():
        label = elem_row[0]
        start = int(elem_row[1])
        stop = int(elem_row[2])
        sensor_data = sensor_df[(sensor_df['Millis'] >= start) & (sensor_df['Millis'] < stop)]

        labels_with_data.append((label, sensor_data))
    return labels_with_data


def read_and_pickle_data():
    # Reads all csv data an splits the recordings into their single equations
    person_folders = [f.path for f in os.scandir(data_folder) if f.is_dir() and (re.match('.*idea.*', f.path) is None)]
    # will contain one element per person,
    # each element will contain all labels and the according sensor data of one person
    all_persons_data = []
    for person_num, person_folder in enumerate(person_folders):
        log(f'Collecting data for person {person_num}...')
        recording_folders = [f.path for f in os.scandir(person_folder) if f.is_dir()]
        person_data = []  # will contain all labels and the according sensor data of one person
        for rec_num, recording_folder in enumerate(recording_folders):
            # read sensor and label data
            label_df = pd.read_csv(recording_folder + '/labels.csv', encoding='utf-8', header=0, delimiter=';')
            sensor_df = pd.read_csv(recording_folder + '/sensor_data.csv', encoding='utf-8', header=0, delimiter=';')

            split_labels_with_data = split_elements(sensor_df, label_df)
            log(f'-- Recording {rec_num}: {len(split_labels_with_data)} equations')
            person_data.extend(split_labels_with_data)

        all_persons_data.append(person_data)

    log('\nSummary:')
    log(f'Persons: {len(all_persons_data)}')
    log(f'Overall equations: {sum([len(person_data) for person_data in all_persons_data])}')
    log('\nPickling the data...')
    pickle.dump(all_persons_data, open(data_folder + pkl_filename, 'wb'))


def unpickle_data():
    with open(data_folder + pkl_filename, 'rb') as f:
        pkl_data = pickle.load(f)
        for person_num, person in enumerate(pkl_data):
            log(f'Person {person_num} has {len(person)} equations')

        log(f"\nSample data for label '{pkl_data[0][0][0]}':\n{pkl_data[0][0][1].head}")

        # Example of splitting 5 adaptation equations from a person:
        person_num = 0
        adaptation_equations = pkl_data[person_num][0:5]
        training_equations = pkl_data[person_num][5:]
        log(
            f'person {person_num}: {len(training_equations)} training equations, {len(adaptation_equations)} adaptation equations')

        # make sure you test your algorithms on unseen writers -> it's about user-independent recognition :)

        return pkl_data


def plot_samples(samples):
    sample = samples[0]
    sample_tp = np.transpose(sample)
    plt.figure(figsize=(2 * 20, (13 + 1) / 2 * 5))
    for i in range(len(sample_tp)):
        plt.subplot(7, 2, i + 1)
        plt.xlabel('Time')
        plt.ylabel(i)
        plt.plot(sample_tp[i])
    plt.show()


# read_and_pickle_data()


def get_samples_labels_per_person(data):
    samples = []
    labels = []
    for person in data:
        person_samples = []
        person_labels = []
        for label, sample in person:
            person_samples.append(sample)
            person_labels.append(label)
        samples.append(person_samples)
        labels.append(person_labels)
    return samples, labels


def get_samples_labels(samples_per_person, labels_per_person):
    samples = []
    labels = []
    for person in samples_per_person:
        for sample in person:
            samples.append(sample)
    for person in labels_per_person:
        for label in person:
            labels.append(label)
    return samples, labels


def numpify_samples(samples):
    for i in range(len(samples)):
        samples[i] = np.array(samples[i].loc[:, 'Acc1 X':'Force'])  # only keep the relevant columns


def remove_hover(samples):
    # TODO: remove bias too!
    for i in range(len(samples)):
        sample = samples[i]
        # taking a lower bound of 0.2N for the force
        j = 0
        while j < len(sample) and sample[j][12] <= 200:
            j += 1

        k = len(sample) - 1
        while k > 0 and sample[k][12] <= 200:
            k -= 1
        samples[i] = sample[j:k + 1]


def avg_len(samples):
    avg = 0
    len_samples = len(samples)
    for i in range(len_samples):
        avg += len(samples[i]) / len_samples
    return avg


def pad_or_resample_samples(samples):
    global DESIRED_SAMPLE_LENGTH
    if DESIRED_SAMPLE_LENGTH != -1:
        desired_len = DESIRED_SAMPLE_LENGTH
    else:
        _avg_len = int(avg_len(samples))
        desired_len = 2 * _avg_len
    for i in range(len(samples)):
        sample = samples[i]
        if len(sample) < desired_len:
            # pad
            padding_len = desired_len - len(sample)
            samples[i] = np.pad(sample, ((0, padding_len), (0, 0)), mode='constant')
        elif len(sample) > desired_len:
            # downsample
            sample_tp = np.transpose(sample)
            x = np.arange(0, len(sample_tp[0]), len(sample_tp[0]) / desired_len)[:desired_len]
            resampled = []
            for k in range(len(sample_tp)):
                xp = np.arange(0, len(sample_tp[k]))
                resampled.append(np.interp(x, xp, sample_tp[k]))
            resampled = np.array(resampled)
            samples[i] = np.transpose(resampled)


def resample_samples(samples):
    for i in range(len(samples)):
        sample = samples[i]
        sample_tp = np.transpose(sample)
        x = np.arange(0, len(sample_tp[0]), len(sample_tp[0]) / DESIRED_SAMPLE_LENGTH)[:DESIRED_SAMPLE_LENGTH]
        resampled = []
        for k in range(len(sample_tp)):
            xp = np.arange(0, len(sample_tp[k]))
            resampled.append(np.interp(x, xp, sample_tp[k]))
            if len(resampled[k]) != DESIRED_SAMPLE_LENGTH:
                raise ValueError(
                    f'expected a sample length of {DESIRED_SAMPLE_LENGTH}, but found length {len(resampled[k])}')
        resampled = np.array(resampled)
        samples[i] = np.transpose(resampled)
        # for k in range(len(xs[i]))


def pad_samples(samples):
    max_len = 0
    for sample in samples:
        max_len = max((max_len, len(sample)))
    print(f'max sample length: {max_len}')

    printed_example = False
    for i in range(len(samples)):
        sample = samples[i]
        padding_len = max_len - len(sample)
        samples[i] = np.pad(sample, ((0, padding_len), (0, 0)), mode='constant')
        if not printed_example:
            print(samples[i])
            printed_example = True


def normalize(xs):
    for i in range(len(xs)):
        xs[i] /= np.array([MAX_ACC_FRONT,
                           MAX_ACC_FRONT,
                           MAX_ACC_FRONT,
                           MAX_ACC_BACK,
                           MAX_ACC_BACK,
                           MAX_ACC_BACK,
                           MAX_GYR,
                           MAX_GYR,
                           MAX_GYR,
                           MAX_MAG,
                           MAX_MAG,
                           MAX_MAG,
                           MAX_FORCE / 2]).reshape(1, 13)
        for j in range(len(xs[i])):
            xs[i][j][12] -= 1
    # return xs


def bounds(xs):
    maxs = []
    mins = []
    for k in range(13):
        maxs.append(0)
        mins.append(0)
    for i in range(len(xs)):
        # per item
        for j in range(len(xs[i])):
            # per timestamp
            for k in range(13):
                x = xs[i][j][k]
                if x > maxs[k]:
                    maxs[k] = x
                if x < mins[k]:
                    mins[k] = x
    return mins, maxs


def log_bounds(xs):
    if LOG:
        log(f'min and max per channel: {list(zip(*bounds(xs)))}')


def make_shuffle(samples, labels):
    indices = list(range(len(samples)))
    random.shuffle(indices)
    return [samples[i] for i in indices], [labels[i] for i in indices]


def make_splits(samples, labels):
    splits = []
    indices = [int(i / 5 * len(samples)) for i in range(6)]
    # for i in range(5):
    log(indices)
    splits_indices = list(zip(indices[:len(indices) - 1], indices[1:]))
    for k in range(len(splits_indices)):
        i, j = splits_indices[k]
        splits.append([samples[i:j], labels[i:j]])
    return splits


def make_folds(splits):
    folds = []
    for k in range(len(splits)):
        samples_train = [sample for i in range(len(splits)) if i != k for sample in splits[i][0]]
        labels_train = [label for i in range(len(splits)) if i != k for label in splits[i][1]]
        samples_test = splits[k][0]
        labels_test = splits[k][1]
        folds.append([samples_train, labels_train, samples_test, labels_test])
    return folds


def file_exists(file_name):
    return os.path.isfile(file_name)


def load_data():
    global DESIRED_SAMPLE_LENGTH
    if file_exists(preprocessed_file):
        with open(preprocessed_file, 'rb') as f:
            (samples, labels) = pickle.load(f)
    else:
        if not file_exists(data_folder + pkl_filename):
            read_and_pickle_data()
        data = unpickle_data()
        #         plot_data(data)
        samples_per_person, labels_per_person = get_samples_labels_per_person(data)
        samples, labels = get_samples_labels(samples_per_person, labels_per_person)

        numpify_samples(samples)
        remove_hover(samples)
        normalize(samples)
        pad_or_resample_samples(samples)
        with open(preprocessed_file, 'wb') as f:
            pickle.dump((samples, labels), f)

    # plot_samples(samples)
    return samples, labels