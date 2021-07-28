import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

LOG = True
DATA_SETS = ['both_dep', 'both_indep', 'lower_dep', 'lower_indep', 'upper_dep', 'upper_indep']
def log(o):
    if LOG:
        print(o)


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


def load_data(data_set):
    with open(f'preprocessed/{data_set}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def get_avg_bounds(folds):
    avg_bounds = None
    for fold in folds:
        _bounds = bounds(fold[0])
        if avg_bounds is None:
            avg_bounds = _bounds
            for i in range(len(avg_bounds)):
                for j in range(len(avg_bounds[i])):
                    avg_bounds[i][j] /= len(folds)
        else:
            for i in range(len(avg_bounds)):
                for j in range(len(avg_bounds[i])):
                    avg_bounds[i][j] += _bounds[i][j] / len(folds)
    return avg_bounds

#%%
def bounds_to_md(min, max):
    res = "| min | max |\n| -------------------- | ------------------- |"
    bounds_zipped = zip(min, max)
    for b_z in bounds_zipped:
        res += f'\n| {b_z[0]} | {b_z[1] } |'
    return res




if __name__ == '__main__':
    if not os.path.isfile('bounds.pkl'):
        data_set_to_bounds = {}
        for data_set in DATA_SETS:
            folds = load_data(data_set)
            avg_bounds = get_avg_bounds(folds)
            data_set_to_bounds[data_set] = avg_bounds
        pickle.dump(data_set_to_bounds, open('bounds.pkl', 'wb'))
    data_set_to_bounds = pickle.load(open('bounds.pkl', 'rb'))
    for data_set in DATA_SETS:
        log(data_set)
        log(bounds_to_md(*data_set_to_bounds[data_set]))
        log('')
        data_set_bounds = data_set_to_bounds[data_set]
        log(len(data_set_bounds[0]))
        log(data_set_bounds[0])
        log(data_set_bounds[1])
        plt.bar(np.arange(len(data_set_bounds[0])), list(map(lambda x: x[1] - x[0], zip(*data_set_bounds))), 0.8,
                data_set_bounds[0])
        plt.title(data_set)
        plt.show()
