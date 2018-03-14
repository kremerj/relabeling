import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_svmlight_file
from sklearn.utils import check_random_state
from imageio import imread
from scipy.misc import imresize
from joblib import Memory


memory = Memory(cachedir='cache', verbose=0)


def corrupt(labels, rho, random_state):
    rng = check_random_state(random_state)
    C = np.array([[1 - rho[0], rho[1]], [rho[0], 1 - rho[1]]])
    return np.array([rng.choice(2, p=C[:, l]) for l in labels.astype('int')])


def generate_dataset(r, dataset, params, scale=True):
    if dataset == 'ad':
        converters = {1558: lambda s: float(s == b'ad.')}
        SampleX = np.genfromtxt('data/uci/ad/ad.data', delimiter=',', usecols=range(1558))
        SampleY = np.genfromtxt('data/uci/ad/ad.data', delimiter=',', usecols=[1558], converters=converters)
        valid_rows = ~np.any(np.isnan(SampleX), axis=1)
        SampleX = SampleX[valid_rows]
        SampleY = SampleY[valid_rows]
    elif dataset == 'a1a':
        SampleX, SampleY = load_svmlight_file("data/libsvm/a1a/a1a.t")
        SampleX = SampleX.toarray()
        SampleY[SampleY == -1] = 0
    elif dataset == 'w1a':
        SampleX, SampleY = load_svmlight_file("data/libsvm/w1a/w1a.t")
        SampleX = SampleX.toarray()
        SampleY[SampleY == -1] = 0
    elif dataset == 'covtype':
        SampleX, SampleY = load_svmlight_file("data/libsvm/covtype/covtype.libsvm.binary")
        SampleX = SampleX.toarray()
        SampleY -= 1
    elif dataset == 'mushrooms':
        SampleX, SampleY = load_svmlight_file("data/libsvm/mushrooms/mushrooms")
        SampleX = SampleX.toarray()
        SampleY -= 1
    elif dataset == 'cod-rna':
        SampleX, SampleY = load_svmlight_file("data/libsvm/cod-rna/cod-rna")
        SampleX = SampleX.toarray()
        SampleY[SampleY == -1] = 0
    elif dataset == 'ijcnn1':
        SampleX, SampleY = load_svmlight_file("data/libsvm/ijcnn1/ijcnn1")
        SampleX = SampleX.toarray()
        SampleY[SampleY == -1] = 0
    elif dataset == 'baidu':
        return generate_deep(r, params)

    # Randomly split into stratified train and test sample
    Sx, Tx, Sy_clean, Ty = train_test_split(
        SampleX,
        SampleY,
        stratify=SampleY,
        train_size=params['n_train'],
        test_size=params['n_test'],
        random_state=r)

    # generate noisy labels
    Sy_noise = corrupt(Sy_clean, params['noise_rate'], random_state=r)

    if scale:
        scaler = StandardScaler()
        Sx = scaler.fit_transform(Sx)
        Tx = scaler.transform(Tx)

    return Sx, Sy_clean, Sy_noise, Tx, Ty


def build_image_data(dataframe, rows=None):
    N = len(dataframe) if rows is None else rows
    w = 227
    h = 227
    d = 3
    output = np.zeros((N, w, h, d), dtype=np.float32)
    for i in range(N):
        im = imresize(imread(dataframe.filename.iloc[i])[:,:,:3].astype(np.float32), (w, h))
        im = im - np.mean(im)
        output[i, ...] = im
    return output


@memory.cache
def generate_deep(r, params):
    classes = [11, 12]  # dress, vest
    data_dir = 'data/baidu/'
    clean_label_file = data_dir + 'annotations/clean_label_kv.txt'
    noisy_label_file = data_dir + 'annotations/noisy_label_kv.txt'
    data_clean = pd.read_csv(clean_label_file, delimiter=' ', names=['filename', 'label'])
    data_noise = pd.read_csv(noisy_label_file, delimiter=' ', names=['filename', 'label'])
    data = pd.merge(data_clean, data_noise, on='filename', how='inner', suffixes=('_clean', '_noise'))
    data = data[data.label_clean.isin(classes) & data.label_noise.isin(classes)]
    data.filename = data_dir + 'images/relabeling/' + data.filename.apply(lambda x: os.path.basename(x))

    data.label_clean = LabelEncoder().fit_transform(data.label_clean)
    data.label_noise = LabelEncoder().fit_transform(data.label_noise)

    indices = range(len(data))
    train_indices, test_indices = train_test_split(indices, train_size=params['n_train'],
                                                   test_size=params['n_test'],
                                                   random_state=r, stratify=data.label_clean)

    Sy_noise = data.label_noise.values[train_indices]
    Sy_clean = data.label_clean.values[train_indices]
    Ty = data.label_clean.values[test_indices]
    data_array = build_image_data(data)
    Sx = data_array[train_indices]
    Tx = data_array[test_indices]
    return Sx, Sy_clean, Sy_noise, Tx, Ty
