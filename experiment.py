import pipeline
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout, Input,Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras.utils import multi_gpu_model
from prettytable import PrettyTable
from pandas import get_dummies
 

def main(feature_extractmethods):
    params = {
        'dim': pipeline.get_directory("dataset/"+feature_extractmethods)[:-1],
        'batch_size': 32,
        'n_classes': 10,
        'n_channels': 3,
        'shuffle': True}
    verbose = 2
    fold_dict = pipeline.get_ids('dataset/metadata/UrbanSound8K.csv')
    t = PrettyTable(['Exp-Num'] + ['Acc', 'Recall', 'F1-Score', 'f1'])
    for i in range (10):
        training_data_partition = []
        testing_data_partition = []

        training_label_partition = []
        testing_label_partition = []

        for j in range(i):
            training_data_partition.extend(fold_dict[j][0])
            training_label_partition.extend(fold_dict[j][1])

        for j in range(i+1, 10):
            training_data_partition.extend(fold_dict[j][0])
            training_label_partition.extend(fold_dict[j][1])

        testing_data_partition.extend(fold_dict[i][0])
        testing_label_partition.extend(fold_dict[i][1])
        test_batchs = len(testing_label_partition) // params['batch_size']
        
        testing_label_partition = testing_label_partition[:test_batchs*params['batch_size']]


        validation_data_partition = training_data_partition[len(training_data_partition) // 10 * 9:]
        training_data_partition = training_data_partition[:len(training_data_partition) // 10 * 9]

        training_generator = pipeline.DataGenerator("dataset/"+feature_extractmethods, training_data_partition, training_label_partition, **params)
        validation_generator = pipeline.DataGenerator("dataset/"+feature_extractmethods, validation_data_partition, training_label_partition, **params)

        test_generator = pipeline.DataGenerator("dataset/"+feature_extractmethods, testing_data_partition, testing_label_partition, **params)

        onehot_testing_labelpartition, label_dict = onehot_encoding(testing_label_partition, params['n_classes'])
        # label_dict = get_label_dict(testing_label_partition, onehot_testing_labelpartition, params['n_classes'])

        model = get_model(params['dim']+(params['n_channels'],), params['n_classes'])
        
        es = EarlyStopping(monitor='val_loss', mode = 'min', verbose = verbose, patience = 15)

        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                    optimizer=tensorflow.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])

        model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=100,
                    verbose = verbose,
                    callbacks=[es],
                    # validation_steps=len(validation_data_partition)//params['batch_size'],
                    workers=1)
        
        yhat_probs = model.predict_generator(test_generator, verbose = verbose) 
        y_predict_classes_onthot = []
        for prob in yhat_probs:
            maxElement = np.where(prob == np.amax(prob))[0][0]
            temp = '['
            temp += '0 ' * (maxElement)
            temp += '1'
            temp += ' 0' * (params['n_classes']-maxElement-1)
            temp += ']'
            y_predict_classes_onthot.append(temp)
   
        yhat_probs = np.squeeze(yhat_probs)
        yhat_classes = np.squeeze(y_predict_classes_onthot)
        testy = np.squeeze(onehot_testing_labelpartition)
        temp = []
        for y in testy:
            temp += [np.array2string(y)]
        testy = temp
        testy = onehot_to_label(testy, label_dict)
        yhat_classes = onehot_to_label(yhat_classes, label_dict)

        accuracy = accuracy_score(testy, yhat_classes)
        precision = precision_score(testy, yhat_classes, average='macro')
        recall = recall_score(testy, yhat_classes, average='macro')
        f1 = f1_score(testy, yhat_classes, average='macro')
        t.add_row([i, accuracy, precision, recall, f1])
        print([i, accuracy, precision, recall, f1])
    print(t)



def get_model(input_shape, numofclasses):
    inputs = Input(shape=input_shape)
    master = inputs

    for i in range (5):
        master = convLayer(min(64, 2 ** (i+4)), master, kernel_size = (5,5), strides = (2,2))


    master = Flatten()(master)
    master = Dense(50, activation = 'relu')(master)
    master = Dropout(0.25)(master)
    master = Dense(20, activation = 'relu')(master)
    master = Dropout(0.25)(master)
    master = Dense(numofclasses, activation = 'softmax')(master)

    model= Model(inputs = inputs, outputs = master)
    return model

def convLayer(filters, previouslayer, kernel_size = (5,5), strides = (2,2)):
    outputs = Conv2D(filters, kernel_size=kernel_size, activation = 'relu', padding = 'same')(previouslayer)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Dropout(0.25)(outputs)
    outputs = AveragePooling2D(pool_size=(2,2), strides = (2,2))(outputs)
    return outputs

def onehot_to_label(y_onehot, label_dict):
    result = []
    for y in y_onehot:
        result += [label_dict[y]]

    return result

def onehot_encoding(Y, num_classes):
    onehot = get_dummies(Y)
    Y_onehot = onehot.values
    label_dict = get_label_dict(Y, Y_onehot, num_classes)
    return Y_onehot, label_dict

def get_label_dict(Y, Y_onehot, num_classes):
    label_dict = dict()
    while num_classes != 0:
        for i in range(len(Y)):
            key = np.array2string(Y_onehot[i])

            if key not in label_dict:
                num_classes -= 1
                label_dict[key] = Y[i]
    return label_dict

if __name__ == "__main__": 
    features = ['mfcc_figs/', 'chroma_figs/', 'spectral_figs/']
    for feature in features:
        main(feature)