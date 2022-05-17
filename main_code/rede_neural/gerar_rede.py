import tensorflow as tf
import numpy as np
import pandas as pd
import os
from time import time


def get_domain_inputs(dataframe):
    # return -> domain = 0 -> binario, 1 -> inteiro, 2 -> continua

    domain = []
    for column in dataframe.columns[:-1]:  # percorre o dataframe por colunas até a penultima coluna
        if len(dataframe[column].unique()) == 2:  # verifica se só há variáveis binárias
            domain.append(0)
        elif np.any(dataframe[column].unique().astype(np.int64) !=
                    dataframe[column].unique().astype(np.float64)):
            domain.append(2)
        else:
            domain.append(1)

    return domain


def remove_integer_vars(data_train, data_test):
    data = data_train.append(data_test)
    domain = get_domain_inputs(data)

    data_train = data_train.to_numpy()
    data_test = data_test.to_numpy()

    removido = 0

    print(domain)

    for index, tipo in enumerate(domain):
        if tipo == 1:
            data_train = np.delete(data_train, index - removido, 1)
            data_test = np.delete(data_test, index - removido, 1)
            removido += 1

    print(len(domain)+1, data_train.shape[1])

    return data_train, data_test


def train_network(dir_path, n_neurons, n_hidden_layers, num_classes):
    data_train = pd.read_csv(os.path.join('../../datasets', dir_path, 'train.csv'))
    data_test = pd.read_csv(os.path.join('../../datasets', dir_path, 'test.csv'))

    print(f'dataset: {dir_path} \ndominio:')
    data_train, data_test = remove_integer_vars(data_train, data_test)

    x_train, y_train = data_train[:, :-1], data_train[:, -1]
    x_test, y_test = data_test[:, :-1], data_test[:, -1]

    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x_train.shape[1]]),
    ])

    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'], )

    model_path = os.path.join('../../datasets', dir_path,
                              f'model_no_int_{n_hidden_layers}layers_{n_neurons}neurons_{dir_path}.h5')

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ck = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

    start = time()
    model.fit(x_train, y_train_ohe, batch_size=4, epochs=100, validation_data=(x_test, y_test_ohe),
              verbose=2, callbacks=[ck, es])

    print(f'Tempo de Treinamento: {time() - start}')

    model = tf.keras.models.load_model(model_path)
    print('Resultado Treinamento')
    model.evaluate(x_train, y_train_ohe, verbose=2)

    print('Resultado Teste')
    model.evaluate(x_test, y_test_ohe, verbose=2)


def setup():
    datasets = [['australian', 2],
                ['auto', 5],
                ['backache', 2],
                ['breast-cancer', 2],
                ['cleve', 2],
                ['cleveland', 5],
                ['glass', 5],
                ['glass2', 2],
                ['heart-statlog', 2],
                ['hepatitis', 2]]

    neurons = [10, 32, 64]

    return [datasets, neurons]


if __name__ == "__main__":
    rede_setup = setup()
    for dataset in rede_setup[0]:
        dir_path = dataset[0]
        num_classes = dataset[1]

        for layers in range(1, 5):

            for n_neurons in rede_setup[1]:
                train_network(dir_path, n_neurons, layers, num_classes)

        print(f'{dir_path} treinado!')
