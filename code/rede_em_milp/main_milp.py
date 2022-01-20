import docplex.mp.model as mp
from cplex import infinity
import numpy as np
import tensorflow as tf
import pandas as pd

import slice
from tjeng import codify_network_tjeng
from fischetti import codify_network_fischetti


def codify_network(modelo_em_tf, dataframe, metodo, num_de_slices=1):
    # modelo_em_tf: um arquivo.h5 lido
    # dataframe: lista lida de um csv
    # metodo: booleano - 0 para "tjeng" - 1 para "fischetti"
    # num_de_slices: inteiro indicando as fatias da rede

    layers = modelo_em_tf.layers
    modelo_em_milp = mp.Model()

    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input)
    '''
    print("bounds_input")
    print(bounds_input)
    '''

    if num_de_slices > 1:
        sliced_bounds_input = slice.slice_bounds(bounds_input, num_de_slices)
        print("sliced_bounds_input")
        print(sliced_bounds_input)
        sliced_bounds_input = np.array(sliced_bounds_input)
        '''
        print("slices")
        print(sliced_bounds_input)
        print("slices, np.array")
        print(np.array(sliced_bounds_input))
        '''
    #     sliced_bounds_input = combine_slices(slices, num_de_slices)
    #
    # input_variables = normaliza_input_variables(modelo_em_milp, domain_input, bounds_input)
    #
    # intermediate_variables = []
    # auxiliary_variables = []
    # decision_variables = []
    #
    # for i in range(len(layers) - 1):
    #     weights = layers[i].get_weights()[0]
    #     intermediate_variables.append(modelo_em_milp.continuous_var_list(weights.shape[1], lb=0, name='y', key_format=f"_{i}_%s"))
    #
    #     if metodo:
    #         auxiliary_variables.append(modelo_em_milp.continuous_var_list(weights.shape[1], lb=0, name='s', key_format=f"_{i}_%s"))
    #     else:
    #         decision_variables.append(
    #             modelo_em_milp.binary_var_list(weights.shape[1], name='a', lb=0, ub=1, key_format=f"_{i}_%s"))
    #
    # output_variables = modelo_em_milp.continuous_var_list(layers[-1].get_weights()[0].shape[1], lb=-infinity, name='o')
    #
    # if not metodo:
    #     modelo_em_milp, output_bounds = codify_network_tjeng(modelo_em_milp, layers, input_variables,
    #                                                          intermediate_variables, decision_variables, output_variables)
    # else:
    #     modelo_em_milp, output_bounds = codify_network_fischetti(modelo_em_milp, layers, input_variables, auxiliary_variables,
    #                                                              intermediate_variables, decision_variables, output_variables)
    #
    # return modelo_em_milp, output_bounds


def normaliza_input_variables(modelo_em_milp, domain_input, bounds_input):
    # modelo_em_milp: um objeto mp.Model -> representa a rede em milp
    # domain_input: vetor de strings -> representa o tipo de intup de cada variável
    # bounds_input: representa

    input_variables = []
    for i in range(len(domain_input)):
        lb, ub = bounds_input[i]
        if domain_input[i] == 'C':
            input_variables.append(modelo_em_milp.continuous_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'I':
            input_variables.append(modelo_em_milp.integer_var(lb=lb, ub=ub, name=f'x_{i}'))
        elif domain_input[i] == 'B':
            input_variables.append(modelo_em_milp.binary_var(name=f'x_{i}'))
    return input_variables


def get_domain_and_bounds_inputs(dataframe):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:  # percorre o dataframe por colunas até a penultima coluna
        if len(dataframe[column].unique()) == 2:  # verifica se só há variáveis binárias
            domain.append('B')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        elif np.any(dataframe[column].unique().astype(np.int64) !=
                    dataframe[column].unique().astype(np.float64)):
            domain.append('C')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        else:
            domain.append('I')
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])

    return domain, bounds


if __name__ == '__main__':
    METODO_TJENG = False
    METODO_FISCHETTI = True
    NUM_DE_SLICES = 2
    path_dir = 'glass'

    # modelo_em_tf = tf.keras.models.load_model(f'datasets/{path_dir}/model_2layers_{path_dir}.h5')

    modelo_em_tf = tf.keras.models.load_model(f'datasets/{path_dir}/teste.h5')

    data_test = pd.read_csv(f'datasets/{path_dir}/test.csv')
    data_train = pd.read_csv(f'datasets/{path_dir}/train.csv')
    data = data_train.append(data_test)
    data = data[['RI', 'Na', 'target']]

    codify_network(modelo_em_tf, data, METODO_TJENG, NUM_DE_SLICES)
    # modelo_em_milp, bounds = codify_network(modelo_em_tf, data, METODO_TJENG, NUM_DE_SLICES)
    # print(modelo_em_milp.export_to_string())
    # print(bounds)

# X ---- E
# x1 == 1 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL
# x1 >= 0 /\ x1 <= 100 /\ x2 == 3 /\ F /\ ~E    INSATISFÁTIVEL -> x1 n é relevante,  SATISFÁTIVEL -> x1 é relevante
'''
print("\n\nSolving modelo_em_tf....\n")

modelo_solve = modelo_em_milp.solve(log_output=True)
print(modelo_em_milp.get_solve_status())
'''
