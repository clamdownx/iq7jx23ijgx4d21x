import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler


def load_data(data_name):
    if data_name == '100leaves':
        data = scipy.io.loadmat(f"data/datasets/{data_name}.mat")
        new_data = {}
        new_data['X'] = np.empty((3, 1), dtype=object)
        new_data['X'][0, 0] = StandardScaler().fit_transform(data['X'][0][0])
        new_data['X'][1, 0] = StandardScaler().fit_transform(data['X'][0][1])
        new_data['X'][2, 0] = StandardScaler().fit_transform(data['X'][0][2])
        new_data['Y'] = data['Y'].flatten()
        return new_data

    if data_name == 'Scene-15':
        data = scipy.io.loadmat(f"data/datasets/{data_name}.mat")
        new_data = {}
        new_data['X'] = np.empty((3, 1), dtype=object)
        new_data['X'][0, 0] = StandardScaler().fit_transform(data['X'][0][0])
        new_data['X'][1, 0] = StandardScaler().fit_transform(data['X'][0][1])
        new_data['X'][2, 0] = StandardScaler().fit_transform(data['X'][0][2])
        new_data['Y'] = data['Y'].flatten()
        return new_data

    if data_name == "Handwritten":
        data = scipy.io.loadmat(f"data/datasets/{data_name}.mat")
        new_data = {}
        new_data['X'] = np.empty((6, 1), dtype=object)
        new_data['X'][0, 0] = StandardScaler().fit_transform(data['X'][0][0])
        new_data['X'][1, 0] = StandardScaler().fit_transform(data['X'][0][1])
        new_data['X'][2, 0] = StandardScaler().fit_transform(data['X'][0][2])
        new_data['X'][3, 0] = StandardScaler().fit_transform(data['X'][0][3])
        new_data['X'][4, 0] = StandardScaler().fit_transform(data['X'][0][4])
        new_data['X'][5, 0] = StandardScaler().fit_transform(data['X'][0][5])
        new_data['Y'] = data['Y'].flatten()
        return new_data

    if data_name == "MSRC-v1":
        data = scipy.io.loadmat(f"data/datasets/{data_name}.mat")
        new_data = {}
        new_data['X'] = np.empty((6, 1), dtype=object)
        new_data['X'][0, 0] = StandardScaler().fit_transform(data['X'][0][0])
        new_data['X'][1, 0] = StandardScaler().fit_transform(data['X'][1][0])
        new_data['X'][2, 0] = StandardScaler().fit_transform(data['X'][2][0])
        new_data['X'][3, 0] = StandardScaler().fit_transform(data['X'][3][0])
        new_data['X'][4, 0] = StandardScaler().fit_transform(data['X'][4][0])
        new_data['X'][5, 0] = StandardScaler().fit_transform(data['X'][5][0])
        new_data['Y'] = data['Y'].flatten()
        return new_data

    if data_name == "LandUse-21":
        data = scipy.io.loadmat(f"data/datasets/{data_name}.mat")
        new_data = {}
        new_data['X'] = np.empty((3, 1), dtype=object)
        new_data['X'][0, 0] = StandardScaler().fit_transform(data['X'][0][0])
        new_data['X'][1, 0] = StandardScaler().fit_transform(data['X'][0][1])
        new_data['X'][2, 0] = StandardScaler().fit_transform(data['X'][0][2])
        new_data['Y'] = data['Y'].flatten()
        return new_data

    raise ValueError(f"Unknown dataset: {data_name}")


def get_multiview_data(data_name):
    data = load_data(data_name)
    X_container = data['X']
    views = [X_container[v][0] for v in range(X_container.shape[0])]
    labels = data['Y'].flatten()
    return views, labels


def missing(views, missing_rate=0.5, seed=42):
    np.random.seed(seed)
    n_views = len(views)
    n_samples = views[0].shape[0]
    missing_views = [v.copy() for v in views]
    missing_mask = np.ones((n_samples, n_views), dtype=int)

    for v in range(n_views):
        n_missing = int(n_samples * missing_rate)
        missing_idx = np.random.choice(n_samples, n_missing, replace=False)
        missing_views[v][missing_idx] = 0.0
        missing_mask[missing_idx, v] = 0

    fully_missing = np.where(~missing_mask.any(axis=1))[0]
    for idx in fully_missing:
        rescue_view = np.random.randint(0, n_views)
        missing_views[rescue_view][idx] = views[rescue_view][idx]
        missing_mask[idx, rescue_view] = 1

    return missing_views, missing_mask