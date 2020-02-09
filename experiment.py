import pipeline
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout, Input,Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import multi_gpu_model
 

def main():
    params = {
        'dim': pipeline.get_directory("dataset/mfcc_figs/")[:-1],
        'batch_size': 64,
        'n_classes': 10,
        'n_channels': 3,
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

        validation_data_partition = training_data_partition[len(training_data_partition) // 10 * 9:]
        training_data_partition = training_data_partition[:len(training_data_partition) // 10 * 9]

        training_generator = pipeline.DataGenerator("dataset/mfcc_figs/", training_data_partition, training_label_partition, **params)
        validation_generator = pipeline.DataGenerator("dataset/mfcc_figs/", validation_data_partition, training_label_partition, **params)

        test_generator = pipeline.DataGenerator("dataset/mfcc_figs/", testing_data_partition, testing_label_partition, **params)

        model = get_model(params['dim']+(params['n_channels'],), params['n_classes'])
        # model.summary()
        # model = multi_gpu_model(model, gpus=2)
        
        es = EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = 15)

        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                    optimizer=tensorflow.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])

        model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=100,
                    workers=6)

def get_model(input_shape, numofclasses):
    inputs = Input(shape=input_shape)
    master = inputs

    for i in range (5):
        master = convLayer(min(64, 2 ** (i+4)), master, kernel_size = (3,3), strides = (1,1))

    master = Flatten()(master)
    master = Dense(50, activation = 'relu')(master)
    master = Dense(20, activation = 'relu')(master)
    master = Dense(numofclasses, activation = 'softmax')(master)

    model= Model(inputs = inputs, outputs = master)
    return model

def convLayer(filters, previouslayer, kernel_size = (5,5), strides = (2,2)):
    outputs = Conv2D(filters, kernel_size=kernel_size, activation = 'relu', padding = 'same')(previouslayer)
    # outputs = BatchNormalization(axis=-1)(outputs)
    outputs = AveragePooling2D(pool_size=(2,2), strides = (2,2))(outputs)
    return outputs

if __name__ == "__main__": 
    main()
    