import numpy as np
import tensorflow as tf
from time import time
from statistics import mean, stdev
import pandas as pd

from main_code.rede_em_milp import main_milp as mm


def copia_modelos(modelos_em_milp):
    modelos_aux = []
    for m in modelos_em_milp:
        modelos_aux.append(m.clone())

    return modelos_aux


def insert_output_constraints_fischetti(mdl, output_variables, network_output, binary_variables):
    variable_output = output_variables[network_output]
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            p = binary_variables[aux_var]
            aux_var += 1
            mdl.add_indicator(p, variable_output <= output, 1)

    return mdl


def insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables, output_bounds):
    variable_output = output_variables[network_output]
    upper_bounds_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:,
                                                            0]  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            z = binary_variables[aux_var]
            mdl.add_constraint(variable_output - output - ub * (1 - z) <= 0)
            aux_var += 1

    return mdl


def get_miminal_explanation(mdl, network_input, network_output, n_classes, method, output_bounds=None):
    assert not (
            not method and output_bounds is None), 'If the method tjeng is chosen, output_bounds must be passed.'

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_constraints = mdl.add_constraints(
        [input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')

    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    if not method:
        mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables,
                                              output_bounds)
    else:
        mdl = insert_output_constraints_fischetti(mdl, output_variables, network_output,
                                                  binary_variables)

    for i in range(len(network_input[0])):
        mdl.remove_constraint(input_constraints[i])

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(input_constraints[i])

    return mdl.find_matching_linear_constraints('input')


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
                ['hepatitis', 2],
                ['spect', 2],
                ['voting', 2]]

    configurations = [5, 10, 20, 40]

    return [datasets, configurations]


def test_get_explanation(models, network_input, network_output, n_classes, method, list_output_bounds=None):
    assert not (
            not method and list_output_bounds is None), 'If the method tjeng is chosen, output_bounds must be passed.'

    mdl = models[0]
    # print(mdl, n_classes, method)
    # print("input")
    # print(network_input)
    # print("output")
    # print(network_output)
    # print("bounds")
    # print(list_output_bounds)

    input_variables = [mdl.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [mdl.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_constraints = mdl.add_constraints(
        [input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
    binary_variables = mdl.binary_var_list(n_classes - 1, name='b')

    mdl.add_constraint(mdl.sum(binary_variables) >= 1)

    if not method:
        mdl = insert_output_constraints_tjeng(mdl, output_variables, network_output, binary_variables,
                                              list_output_bounds[0])
    else:
        mdl = insert_output_constraints_fischetti(mdl, output_variables, network_output,
                                                  binary_variables)

    for i in range(len(network_input[0])):
        mdl.remove_constraint(input_constraints[i])

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(input_constraints[i])

    return mdl.find_matching_linear_constraints('input')


def main():
    METODO_TJENG = False
    METODO_FISCHETTI = True
    NUM_DE_SLICES = 1
    path_dir = 'glass'
    n_classes = 5

    modelo_em_tf = tf.keras.models.load_model(f'../../datasets/{path_dir}/teste.h5')

    data_test = pd.read_csv(f'../../datasets/{path_dir}/test.csv')
    data_train = pd.read_csv(f'../../datasets/{path_dir}/train.csv')
    data = data_train.append(data_test)
    data = data[['RI', 'Na', 'target']]
    data_aux = data.to_numpy()

    lista_de_modelos_em_milp, lista_de_bounds = mm.codify_network(modelo_em_tf, data, METODO_TJENG, NUM_DE_SLICES)
    for i in range(data_aux.shape[0]):
        print(f'dado: {i}')
        network_input = data_aux[i, :-1]

        network_input = tf.reshape(tf.constant(network_input), (1, -1))
        network_output = modelo_em_tf.predict(tf.constant(network_input))[0]
        network_output = tf.argmax(network_output)

        mdl_aux = copia_modelos(lista_de_modelos_em_milp)
        start = time()

        explanation = test_get_explanation(mdl_aux, network_input, network_output, n_classes=n_classes,
                                           method=METODO_TJENG, list_output_bounds=lista_de_bounds)

        print(explanation)
        print(time() - start)

        print()


if __name__ == '__main__':
    main()

    # METODO_TJENG = False
    # METODO_FISCHETTI = True
    #
    # datasets, configurations = setup()
    #
    # for dataset in datasets:
    #     dir_path = dataset[0]
    #     n_classes = dataset[1]
    #
    #     data_test = pd.read_csv(f'../../datasets/{dir_path}/test.csv')
    #     data_train = pd.read_csv(f'../../datasets/{dir_path}/train.csv')
    #
    #     data = data_train.append(data_test)
    #     print(data)
    #     data_aux = data.to_numpy()
    #     print(data_aux)
    #     for neurons in configurations:  # 5 10 20 40
    #         print(dataset, neurons)
    #
    #         model_path_1layer = f'../../datasets/{dir_path}/model_1layers_{neurons}neurons_{dir_path}.h5'
    #         model_1layer = tf.keras.models.load_model(model_path_1layer)
    #
    #         model_path_2layers = f'../../datasets/{dir_path}/model_2layers_{neurons}neurons_{dir_path}.h5'
    #         model_2layers = tf.keras.models.load_model(model_path_2layers)
    #
    #         for sliced in range(1, 4):  # 1, 2, 3
    #             print(f'slices: {sliced}')
    #             for metodo in range(1, -1, -1):  # 1, 0
    #                 print(f'metodo 1 = F, 0 = T: {metodo}')
    #                 lista_de_modelos_em_milp_1layer, lista_de_bounds_1layer = mm.codify_network(model_1layer,
    #                                                                                             data, metodo, sliced)
    #                 lista_de_modelos_em_milp_2layers, lista_de_bounds_2layers = mm.codify_network(model_2layers,
    #                                                                                               data, metodo, sliced)
    #                 for i in range(data_aux.shape[0]):
    #                     print(f'dado: {i}')
    #                     network_input = data_aux[i, :-1]
    #
    #                     network_input_1layer = tf.reshape(tf.constant(network_input), (1, -1))
    #                     network_output_1layer = model_1layer.predict(tf.constant(network_input_1layer))[0]
    #                     network_output_1layer = tf.argmax(network_output_1layer)
    #
    #                     network_input_2layers = tf.reshape(tf.constant(network_input), (1, -1))
    #                     network_output_2layers = model_2layer.predict(tf.constant(network_input_2layers))[0]
    #                     network_output_2layers = tf.argmax(network_output_2layers)
    #
    #                     mdl_1layer_aux = copia_modelos(lista_de_modelos_em_milp_1layer)
    #                     mdl_2layers_aux = copia_modelos(lista_de_modelos_em_milp_2layers)
    #                     start = time()
    #
    #                     explanation_1layer = get_explanation(mdl_1layer_aux, network_input_1layer,
    #                                                          network_output_1layer,
    #                                                          n_classes=n_classes, method=metodo,
    #                                                          output_bounds=lista_de_bounds_1layer)
    #
    #                     # print(explanation_1layer)
    #                     print(time() - start)
    #                     explanation_2layers = get_explanation(mdl_2layers_aux, network_input_2layers,
    #                                                           network_output_2layers,
    #                                                           n_classes=n_classes, method=metodo,
    #                                                           output_bounds=lista_de_bounds_2layers)
    #
    #                     # print(explanation_2layers)
