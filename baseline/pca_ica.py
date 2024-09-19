from sklearn.decomposition import PCA, FastICA
import numpy as np
import numpy as np
import pickle


zim_dims = 50

def train_fast_ica(start_time, end_time):
    path =f"./compare_model/FastICA_whole.pkl"
    train_data = np.load(f"./new_data/train/whole.npy")[:, start_time:end_time]
    with open(path, 'wb') as f:
        ica = FastICA(n_components=zim_dims)
        ica.fit(train_data)
        pickle.dump(ica, f)

def train_pca(start_time, end_time):
    path =f"./compare_model/PCA_whole.pkl"
    train_data = np.load(f"./train/whole.npy")[:, start_time:end_time]
    with open(path, 'wb') as f:
        pca = PCA(n_components=zim_dims)
        pca.fit(train_data)
        pickle.dump(pca, f)

if __name__ == "__main__":
    train_fast_ica()
    train_pca()
    
