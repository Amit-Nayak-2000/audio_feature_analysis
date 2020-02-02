import pipeline
from keras.models import Sequential
import numpy as np 

def main():
    params = {
        'dim': pipeline.get_directory("dataset/mfcc_figs/"),
        'batch_size': 64,
        'n_classes': 10,
        'n_channels': 4,
        'shuffle': True}

    fold_dict = pipeline.get_ids('dataset/metadata/UrbanSound8K.csv')

    for i in range (10):
        training_data_partition = []
        testing_data_partition = []

        training_label_partition = []
        testing_label_partition = []

        for j in range(i):
            training_data_partition.extend(fold_dict[i][0])
            training_label_partition.extend(fold_dict[i][1])

        for j in range(i+1, 10):
            training_data_partition.extend(fold_dict[i][0])
            training_label_partition.extend(fold_dict[i][1])

        testing_data_partition.extend(fold_dict[i][0])
        testing_label_partition.extend(fold_dict[i][1])

        training_generator = pipeline.DataGenerator("dataset/mfcc_figs/", training_data_partition, training_label_partition, **params)
        validation_generator = pipeline.DataGenerator("dataset/mfcc_figs/", testing_data_partition, testing_label_partition, **params)





