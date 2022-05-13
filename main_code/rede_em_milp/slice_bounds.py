import numpy as np


def slice_continous_var_list(bounds_input, domain, limit_of_sliced_vars):
    # bounds_input: lista de listas de pares de valores -> máximo e mínimo dos inputs
    # domain: lista de inteiros -> números representando o domínio dos bounds_input
    # limit_of_sliced_vars: inteiro -> número máximo de varíavies a serem fatiadas
    #
    # return -> lista com bounds input duplicada n vezes em variáveis contínuas,
    #           na qual n é o valor de limit_of_sliced_vars
    
    limit_aux = 0
    list_var_to_slice = []
    len_domain_minus1 = len(domain) - 1
    for index, tipo in enumerate(domain[::-1]):
        if tipo == 2 and limit_aux < limit_of_sliced_vars:
            list_var_to_slice.append(len_domain_minus1 - index)
            limit_aux += 1
    list_bounds_input = [bounds_input]
    if limit_of_sliced_vars > limit_aux:
        raise Exception('Não há variáveis suficientes para fatiar')

    for index in list_var_to_slice:
        list_bounds_input = slice_bounds_by_var(list_bounds_input, index)

    print("num de redes:", 2**len(list_var_to_slice))

    return np.array(list_bounds_input), [2**len(list_var_to_slice), list_var_to_slice[::-1]]


def slice_bounds_by_var(list_bounds_input, index_var):
    list_bounds_input_aux = []

    for bounds_input in list_bounds_input:

        amplitude_relativa = (bounds_input[index_var][1] - bounds_input[index_var][0]) / 2

        # primeira fatia
        bound_input_aux = bounds_input[index_var].copy()
        bound_input_aux[1] -= amplitude_relativa
        bounds_input[index_var] = bound_input_aux.copy()
        list_bounds_input_aux.append(bounds_input.copy())

        # segunda fatia
        bound_input_aux = bounds_input[index_var].copy()
        bound_input_aux[0] += amplitude_relativa
        bound_input_aux[1] += amplitude_relativa
        bounds_input[index_var] = bound_input_aux.copy()
        list_bounds_input_aux.append(bounds_input.copy())

    return list_bounds_input_aux
