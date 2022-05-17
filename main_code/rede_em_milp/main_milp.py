from time import time

import docplex.mp.model as mp
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd
from main_code.rede_em_milp import slice_bounds as sb

from main_code.rede_em_milp import tjeng
from main_code.rede_em_milp import fischetti


def codify_network(modelo_em_tf, dataframe, metodo, num_de_sliced_var=0):
    # modelo_em_tf: um arquivo.h5 lido
    # dataframe: lista - lida de um csv
    # metodo: booleano - 0 para "tjeng" - 1 para "fischetti"
    # num_de_sliced_var: inteiro indicando as variáveis fatiáveis da rede

    layers = modelo_em_tf.layers
    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)

    # print(domain_input)

    if num_de_sliced_var > 0:
        sliced_bounds_input, list_var_sliced = sb.slice_continous_var_list(bounds_input, domain_input, num_de_sliced_var)
        num_redes = list_var_sliced[0]

        # print(f'Redes: {num_redes}')

    else:
        sliced_bounds_input = [bounds_input]
        sliced_bounds_input = np.array(sliced_bounds_input)
        list_var_sliced = [1, []]
        num_redes = 1

    lista_de_milp_models = instancia_mp_models(num_redes)

    lista_de_modelos_em_milp = []
    lista_de_output_bounds = []

    for index, milp_model in enumerate(lista_de_milp_models):

        input_variables = atribui_input_variables(milp_model, domain_input, sliced_bounds_input[index])

        intermediate_variables = []
        auxiliary_variables = []
        decision_variables = []
        for i in range(len(layers) - 1):
            weights = layers[i].get_weights()[0]

            intermediate_variables.append(
                milp_model.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))

            if metodo:
                auxiliary_variables.append(
                    milp_model.continuous_var_list(weights.shape[1], lb=0, name='s', key_format=f"_{i}_%s"))

            decision_variables.append(
                milp_model.binary_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))

        output_variables = milp_model.continuous_var_list(layers[-1].get_weights()[0].shape[1], lb=-infinity, name='o')

        if not metodo:
            modelo_em_milp, output_bounds = tjeng.codify_network_tjeng(milp_model, layers,
                                                                       input_variables,
                                                                       intermediate_variables,
                                                                       decision_variables,
                                                                       output_variables)
        else:
            modelo_em_milp, output_bounds = fischetti.codify_network_fischetti(milp_model, layers,
                                                                               input_variables,
                                                                               auxiliary_variables,
                                                                               intermediate_variables,
                                                                               decision_variables,
                                                                               output_variables)

        lista_de_modelos_em_milp.append(modelo_em_milp)
        lista_de_output_bounds.append(output_bounds)

        if index > 5000:
            raise Exception("Muitas redes, talvez eu esteja travando")

    return lista_de_milp_models, [sliced_bounds_input, lista_de_output_bounds, list_var_sliced[1]]


def atribui_input_variables(modelo_em_milp, domain_input, bounds_input):
    # modelo_em_milp: um objeto mp.Model -> representa a rede em milp
    # domain_input: vetor de strings -> representa o tipo de intup de cada variável
    # bounds_input: vetor com pares de valores -> representa os valores máximo e mínimo de cada variável

    input_variables = []
    for i in range(len(domain_input)):
        lb, ub = bounds_input[i]
        if domain_input[i] == 2:
            input_variables.append(modelo_em_milp.continuous_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 1:
            input_variables.append(modelo_em_milp.integer_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 0:
            input_variables.append(modelo_em_milp.binary_var(name=f'x_{i}'))
    return input_variables


def instancia_mp_models(num_de_modelos):
    # return -> uma lista de mp.Model

    lista_de_mp_models = []
    for _ in range(num_de_modelos):
        m = mp.Model()
        lista_de_mp_models.append(m)
    return lista_de_mp_models


def get_domain_and_bounds_inputs(dataframe):
    # As varíaveis inteiras não serão mais utilizadas nesse experimento.

    # return -> duas listas com os valores máximo e mínimo de domínio e limites de entrada:
    #           domain = 0 -> binario, 1 -> inteiro, 2 -> continua
    #           bounds = [[min, max],...]

    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:  # percorre o dataframe por colunas até a penultima coluna
        if len(dataframe[column].unique()) == 2:  # verifica se só há variáveis binárias
            domain.append(0)
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        elif np.any(dataframe[column].unique().astype(np.int64) !=
                    dataframe[column].unique().astype(np.float64)):
            domain.append(2)
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        else:
            continue
            # domain.append(1)
            # bound_inf = dataframe[column].min()
            # bound_sup = dataframe[column].max()
            # bounds.append([bound_inf, bound_sup])

    return domain, bounds


if __name__ == '__main__':
    METODO_TJENG = False
    METODO_FISCHETTI = True
    NUM_SLICED_VARS = 0
    path_dir = 'australian'

    modelo_em_tf = tf.keras.models.load_model(f'../../datasets/{path_dir}/model_no_int_1layers_10neurons_{path_dir}.h5')

    # modelo_em_tf = tf.keras.models.load_model(f'../../datasets/{path_dir}/teste.h5')

    data_test = pd.read_csv(f'../../datasets/{path_dir}/test.csv')
    data_train = pd.read_csv(f'../../datasets/{path_dir}/train.csv')

    data = data_train.append(data_test)

    # data = data[['RI', 'Na', 'target']]

    lista_de_modelos_em_milp, lista_de_bounds = codify_network(modelo_em_tf, data, METODO_FISCHETTI, NUM_SLICED_VARS)

    for i in range(len(lista_de_modelos_em_milp)):
        # print(lista_de_modelos_em_milp[i].export_to_string())
        print(lista_de_bounds[0][i])

# X ---- E
# x1 == 1 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL
# x1 >= 0 /\ x1 <= 100 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL -> x1 nao é relevante,  SATISFÁTIVEL -> x1 é relevante
'''
print("\n\nSolving modelo_em_tf....\n")

modelo_solve = modelo_em_milp.solve(log_output=True)
print(modelo_em_milp.get_solve_status())
'''
