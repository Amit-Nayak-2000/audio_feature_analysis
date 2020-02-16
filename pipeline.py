import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import os
import pandas as pd
import numpy as np 

def main():
    master_shape_mfcc = get_directory("dataset/mfcc_figs/")
    # master_shape_chroma = get_directory("dataset/chroma_figs/")
    # master_shape_spectral = get_directory("dataset/spectral_figs/")
    # if master_shape_mfcc == master_shape_chroma and master_shape_mfcc == master_shape_spectral:
    #     print('success')
    get_ids('dataset/metadata/UrbanSound8K.csv')


def get_directory(location):
    file_list = os.listdir(location)
    for i in range(len(file_list)):
        if file_list[i].endswith(".png"):
            img_location = os.path.join(location, file_list[i])
            img = load_img(img_location)
            img_array = img_to_array(img)
            if i == 0:
                master_shape = img_array.shape
                return master_shape
            else:
                if img_array.shape != master_shape:
                    print("different shape")
                    return 0

    return master_shape 

def get_ids(location):

    metadata = pd.read_csv(location)
    fold_dict = {}
    for index, row in metadata.iterrows():
        fold_dict.setdefault(row["fold"]-1, [[],[]])
        fold_dict[row['fold']-1][0].append(row['slice_file_name'].replace('.wav', '.png'))
        fold_dict[row['fold']-1][1].append(row['classID']-1)
    # for k in (fold_dict.keys()):
    #     print(k)

    return fold_dict

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_location, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.file_location = file_location
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = img_to_array(load_img(self.file_location + ID))
            
            # Store class
            y[i] = ID.split('-')[1]

        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)      

if __name__ == "__main__": 
    main()