import numpy as np
import tensorflow as tf
from time import time
from statistics import mean, stdev
import pandas as pd

from main_code.rede_neural import gerar_rede as gr
from main_code.rede_em_milp import main_milp as mm


# Métodos já existentes


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


# Métodos com novas abordagens


def configura_rede(model, network_input, network_output, n_classes, method, output_bounds, list_irr_var):
    input_variables = [model.get_var_by_name(f'x_{i}') for i in range(len(network_input[0]))]
    output_variables = [model.get_var_by_name(f'o_{i}') for i in range(n_classes)]
    input_constraints = model.add_constraints(
        [input_variables[i] == feature.numpy() for i, feature in enumerate(network_input[0])], names='input')
    binary_variables = model.binary_var_list(n_classes - 1, name='b')

    model.add_constraint(model.sum(binary_variables) >= 1)

    if not method:
        insert_output_constraints_tjeng(model, output_variables, network_output, binary_variables,
                                        output_bounds)
    else:
        insert_output_constraints_fischetti(model, output_variables, network_output,
                                            binary_variables)
    for i in list_irr_var:
        model.remove_constraint(input_constraints[i])
    return input_constraints


def get_explanation_sliced(list_models, list_var_sliced, list_output_bounds, config_data, n_classes):

    # desencapsulamento
    ponteiro_rede_pivo = config_data[0]
    method = config_data[1]
    network_input = config_data[2]
    network_output = config_data[3]

    # vetores
    copy_list_models = []
    list_index_models_uteis = []
    list_input_constraints = []
    index_var_irrelevantes = []

    # atribuição inicial de vetores
    copy_list_models.append(list_models[ponteiro_rede_pivo].clone())
    list_index_models_uteis.append(ponteiro_rede_pivo)
    list_input_constraints.append(configura_rede(copy_list_models[0], network_input, network_output,
                                                 n_classes, method, list_output_bounds[ponteiro_rede_pivo], []))

    for i in range(len(network_input[0])):
        index_var_irrelevantes.append(i)
        notvazio = False

        # vetores auxiliares
        copy_list_models_aux = []
        list_index_models_uteis_pivo_aux = []
        list_input_constraints_pivo_aux = []

        for index, rede in enumerate(copy_list_models):
            rede.remove_constraint(list_input_constraints[index][i])
            rede.solve(log_output=False)

            if rede.solution is not None:

                index_var_irrelevantes.remove(i)

                for j in range(index + 1):
                    copy_list_models[j].add_constraint(list_input_constraints[j][i])
                break

        if i not in list_var_sliced or i not in index_var_irrelevantes:
            continue

        for ponteiro in list_index_models_uteis:
            inversa = procura_rede_inversa_by_var(list_var_sliced.index(i), ponteiro)
            copy = list_models[inversa].clone()
            output_bounds = list_output_bounds[inversa]
            input_constraints = configura_rede(copy, network_input, network_output, n_classes, method,
                                               output_bounds, index_var_irrelevantes)

            copy.solve()

            if copy.solution is None:
                copy_list_models_aux.append(copy)
                list_index_models_uteis_pivo_aux.append(inversa)
                list_input_constraints_pivo_aux.append(input_constraints)
                continue
            else:
                index_var_irrelevantes.remove(i)
                for j, rede in enumerate(copy_list_models):
                    rede.add_constraint(list_input_constraints[j][i])
                notvazio = True
                break

        if notvazio:
            continue

        copy_list_models.extend(copy_list_models_aux)
        list_index_models_uteis.extend(list_index_models_uteis_pivo_aux)
        list_input_constraints.extend(list_input_constraints_pivo_aux)

    return copy_list_models[0].find_matching_linear_constraints('input')


def procura_index_rede_pivo(list_input_bounds, list_var_sliced, network_input):
    index_pivo = 0

    for index_enum, index_var in enumerate(list_var_sliced):
        if list_input_bounds[index_pivo][index_var][1] < network_input[index_var]:
            index_pivo += 2 ** index_enum

    return index_pivo


def procura_rede_inversa_by_var(index_var, index_rede_pivo):
    index_var = 2 ** index_var

    if (int(index_rede_pivo / index_var)) % 2:
        return index_rede_pivo - index_var
    else:
        return index_rede_pivo + index_var


# Rotinas de testes


def setup():
    datasets = [['australian', 2],
                ['auto', 5],
                ['backache', 2],
                ['cleve', 2],
                ['cleveland', 5],
                ['glass', 5],
                ['glass2', 2],
                ['heart-statlog', 2],
                ['hepatitis', 2]]

    configurations = [8, 16]

    return [datasets, configurations]


def rotina_1():
    rede_setup = setup()

    for dataset in rede_setup[0]:
        df = {
            'camadas': [],
            'neurônios': [],
            'slices': [],
            'ub_y<=0': [],
            'lb_y>=0': [],
            'sem_simplificação': [],
            'total': [],
            'tempo': []
        }
        start = time()
        dir_path = dataset[0]

        data_test = pd.read_csv(f'../../datasets/{dir_path}/test.csv')
        data_train = pd.read_csv(f'../../datasets/{dir_path}/train.csv')

        data = data_train.append(data_test)
        print(dir_path)

        for layers in range(1, 4):

            for n_neurons in rede_setup[1]:
                if n_neurons >= 32:
                    continue

                start3 = time()
                for slices in range(4):
                    start2 = time()

                    modelo_em_tf = tf.keras.models.load_model(
                        f'../../datasets/{dir_path}/model_no_int_{layers}layers_{n_neurons}neurons_{dir_path}.h5')
                    modelo, results = mm.codify_network(modelo_em_tf, data, 0, slices)

                    df['camadas'].append(layers)
                    df['neurônios'].append(n_neurons)
                    df['slices'].append(slices)
                    df['ub_y<=0'].append(results[0])
                    df['lb_y>=0'].append(results[1])
                    df['sem_simplificação'].append(results[2])
                    df['total'].append(results[0] + results[1] + results[2])
                    df['tempo'].append(time() - start2)

                print(f'{layers}layer(s) {n_neurons}neurons 0 a 3 slices concluída', time() - start3)

        df = pd.DataFrame(df)
        print(df)
        print(f'{dir_path} codificado! tempo: {time() - start}')
        df.to_csv(f'{dir_path}_r1.csv')


def main():
    # METODO_TJENG = False
    # METODO_FISCHETTI = True
    NUM_SLICED_VARS = 2
    path_dir = 'glass'
    n_classes = 5

    # modelo_em_tf = tf.keras.models.load_model(f'../../datasets/{path_dir}/model_2layers_40neurons_{path_dir}.h5')
    #
    # # modelo_em_tf = tf.keras.models.load_model(f'../../datasets/{path_dir}/teste.h5')
    #
    # data_test = pd.read_csv(f'../../datasets/{path_dir}/test.csv')
    # data_train = pd.read_csv(f'../../datasets/{path_dir}/train.csv')
    # data = data_train.append(data_test)
    # # data = data[['RI', 'Na', 'target']]
    # data_aux = data.to_numpy()
    #
    # lista_de_modelos_em_milp, lista_de_bounds = mm.codify_network(modelo_em_tf, data, METODO_FISCHETTI, NUM_SLICED_VARS)
    #
    # list_input_bounds = lista_de_bounds[0]
    # list_output_bounds = lista_de_bounds[1]
    # list_vars_sliced = lista_de_bounds[2]
    #
    # # print(np.array(list_input_bounds))
    # count = 0
    # times = 0
    # for i in range(data_aux.shape[0]):
    #     print(f'dado: {i}')
    #
    #     start = time()
    #     network_input = data_aux[i, :-1]
    #
    #     index_rede_pivo = procura_index_rede_pivo(list_input_bounds, list_vars_sliced, network_input)
    #
    #     # print(network_input)
    #     # print(list_input_bounds[index_rede_pivo])
    #
    #     network_input = tf.reshape(tf.constant(network_input), (1, -1))
    #     network_output = modelo_em_tf.predict(tf.constant(network_input))[0]
    #     network_output = tf.argmax(network_output)
    #
    #     config = [index_rede_pivo, n_classes, METODO_FISCHETTI]
    #
    #     if NUM_SLICED_VARS == 0:
    #         mdl_aux = lista_de_modelos_em_milp[0].clone()
    #         explanation = get_miminal_explanation(mdl_aux, network_input, network_output, n_classes, METODO_FISCHETTI,
    #                                               list_output_bounds[0])
    #
    #     else:
    #         explanation = test_get_explanation(lista_de_modelos_em_milp, config, list_vars_sliced,
    #                                            network_input, network_output, list_output_bounds)
    #
    #     # print(explanation[1][3])
    #     # count += explanation[1][3]
    #     print(time() - start)
    #     times += time() - start
    #
    #     print()
    # print(count/205)
    # print(times)


def rotina_2():
    rede_setup = setup()

    for dataset in rede_setup[0]:
        df = {
            'camadas': [],
            'neurônios': [],
            'slices': [],
            'tamanho_max': [],
            'tamanho_min': [],
            'tamanho_médio': [],
            'tamanho_desvio_padrão': [],
            'tempo_max': [],
            'tempo_min': [],
            'tempo_médio': [],
            'tempo_desvio_padrão': [],
            'tempo_total': []
        }
        start1 = time()
        dir_path = dataset[0]
        n_classes = dataset[1]

        data_test = pd.read_csv(f'../../datasets/{dir_path}/test.csv')
        data_train = pd.read_csv(f'../../datasets/{dir_path}/train.csv')
        data = data_train.append(data_test)
        data_train, data_test = gr.remove_integer_vars(data_train, data_test)
        data_aux = np.append(data_test, data_test, 0)
        # print(data_aux)
        print(dir_path)

        for layers in range(1, 5):

            for n_neurons in rede_setup[1]:
                start2 = time()

                for slices in range(4):
                    start3 = time()

                    # codificação milp
                    modelo_em_tf = tf.keras.models.load_model(
                        f'../../datasets/{dir_path}/model_no_int_{layers}layers_{n_neurons}neurons_{dir_path}.h5')
                    modelo, results = mm.codify_network(modelo_em_tf, data, 0, slices)
                    milp_models = modelo[0]
                    list_bounds_input = modelo[1]
                    list_output_bounds = modelo[2]
                    list_vars_sliced = modelo[3]

                    # variáveis de resultado
                    tamanhos = []
                    times = []

                    for i in range(data_aux.shape[0]):
                        start4 = time()
                        network_input = data_aux[i, :-1]

                        index_rede_pivo = procura_index_rede_pivo(list_bounds_input, list_vars_sliced, network_input)

                        network_input = tf.reshape(tf.constant(network_input), (1, -1))
                        network_output = modelo_em_tf.predict(tf.constant(network_input))[0]
                        network_output = tf.argmax(network_output)

                        config_data = [index_rede_pivo, 0, network_input, network_output]

                        if slices == 0:
                            mdl_aux = milp_models[0].clone()
                            explanation = get_miminal_explanation(mdl_aux, network_input, network_output, n_classes,
                                                                  0, list_output_bounds[0])

                        else:
                            explanation = get_explanation_sliced(milp_models, list_vars_sliced, list_output_bounds,
                                                                 config_data, n_classes)

                        # print(f'{layers} layer(s) {n_neurons} neurons {slices} slices dado {i} concluído',
                        #       time() - start4)
                        # print(explanation)

                        times.append(time() - start4)
                        tamanhos.append(len(explanation))

                    df['camadas'].append(layers)
                    df['neurônios'].append(n_neurons)
                    df['slices'].append(slices)
                    df['tamanho_max'].append(max(tamanhos))
                    df['tamanho_min'].append(min(tamanhos))
                    df['tamanho_médio'].append(mean(tamanhos))
                    df['tamanho_desvio_padrão'].append(stdev(tamanhos))
                    df['tempo_max'].append(max(times))
                    df['tempo_min'].append(min(times))
                    df['tempo_médio'].append(mean(times))
                    df['tempo_desvio_padrão'].append(stdev(times))
                    df['tempo_total'].append(time() - start3)

                    print(f'{layers} layer(s) {n_neurons} neurons {slices} slices concluído', time() - start3)

        df = pd.DataFrame(df)
        print(df)
        print(f'{dir_path} explicado! tempo: {time() - start1}')
        df.to_csv(f'{dir_path}_r2.csv')

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


if __name__ == '__main__':
    rotina_2()
