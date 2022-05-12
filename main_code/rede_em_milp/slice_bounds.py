import numpy as np


def slice_continous_var_list(bounds_input, domain, limit_of_sliced_vars):
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

    return list_bounds_input, [2**len(list_var_to_slice), list_var_to_slice[::-1]]


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


'''
    Abaixo há funções para auxiliar na lógica geral das funções acima 
    (Estas funções não estão sendo utilizadas e são obsoletas)
'''


def slice_bounds_all(bounds_input, num_de_sets):
    if num_de_sets < 2 or num_de_sets > 4:
        return [bounds_input], 1

    lista_de_bounds_input = []
    len_bounds_input = 0
    for linha in bounds_input:
        len_bounds_input += 1
        amplitude_relativa = (linha[1] - linha[0]) / num_de_sets
        slices = []
        for i in range(num_de_sets):
            slices.append([linha[0] + amplitude_relativa * i, linha[0] + amplitude_relativa * (i + 1)])
        lista_de_bounds_input.append(slices)

    return combine_sliced_bounds_all(lista_de_bounds_input, len_bounds_input, num_de_sets)


def combine_sliced_bounds_all(slices, num_de_variaveis, num_de_slices):
    # slices: lista com pares de valores -> representa os limites
    # superior e inferior de cada variável fatiada
    # num_de_slices: inteiro -> representa o número de fatias da variável slices

    sliced_bounds_input = []
    num_de_arranjos = num_de_slices ** num_de_variaveis
    # num_de_arranjos: é um inteiro encontrado
    # a partir do número de fatias elevedado ao número de variáveis

    for i in range(num_de_arranjos):
        sliced_aux = []
        for j in range(num_de_variaveis):
            variavel = num_de_variaveis - (j + 1)
            indice = int(i / num_de_slices ** variavel)
            sliced_aux.append(slices[j][indice % num_de_slices])
        sliced_bounds_input.append(sliced_aux)

    return np.array(sliced_bounds_input), num_de_arranjos


def slice_bounds_continous(bounds_input, domain_input, num_de_sets):
    # Esta função realiza slices apenas em variáveis contínuas

    if num_de_sets < 2:
        return [bounds_input], len(domain_input)

    lista_de_bounds_input = []
    contador_bounds_input = 0
    variaveis_fatiadas = 0
    for linha in bounds_input:
        # se a variável não for contínua, não será fatiada

        if domain_input[contador_bounds_input] != 2 : # or variaveis_fatiadas >= num_vars
            lista_de_bounds_input.append([linha])
            contador_bounds_input += 1
            continue

        slices = []
        amplitude_relativa = (linha[1] - linha[0]) / num_de_sets
        for i in range(num_de_sets):
            slices.append([linha[0] + amplitude_relativa * i, linha[0] + amplitude_relativa * (i + 1)])
        lista_de_bounds_input.append(slices)
        contador_bounds_input += 1
        variaveis_fatiadas += 1

    return lista_de_bounds_input, contador_bounds_input


def combine_sliced_bounds_continous(slices, domain_input, num_de_variaveis, num_de_slices):
    # slices: lista com pares de valores -> representa os limites
    # superior e inferior de cada variável fatiada
    # num_de_slices: inteiro -> representa o número de fatias da variável slices

    if num_de_slices < 2:
        return np.array(slices), 1

    sliced_bounds_input = []
    num_de_arranjos = num_de_slices ** num_de_variaveis
    # num_de_arranjos: é um inteiro encontrado
    # a partir do número de fatias elevedado ao número de variáveis

    num_fatias = 0

    for i in range(num_de_arranjos):
        sliced_aux = []
        quebra = False
        for j in range(num_de_variaveis):
            indice = (int(i / num_de_slices ** j)) % num_de_slices
            if (domain_input[j] != 2 and indice != 0) : # or num_fatias >= num_vars
                num_de_arranjos -= 1
                quebra = True
                break

            sliced_aux.append(slices[j][indice])
        if not quebra:
            num_fatias += 1
            sliced_bounds_input.append(sliced_aux)

    return np.array(sliced_bounds_input), num_de_arranjos


def converter_para_maximo_da_base(num_de_digitos, base):
    # num_decimal: inteiro -> representa o número decimal
    # base: inteiro -> representa a base para o número ser convertido
    # retorna uma lista com os digitos na base exigida com valor máximo

    maximo_da_base = []

    for i in range(num_de_digitos):
        maximo_da_base.append(base - 1)

    return maximo_da_base


def proximo(digitos, base):
    digitos_em_zero = []
    for i in reversed(range(len(digitos))):
        digitos_em_zero.append(0)
        if 0 < digitos[i] < base:
            digitos[i] -= 1
            return digitos
        elif digitos[i] == 0:
            digitos[i] = base - 1
            continue
        else:
            print("Erro: conversão errada")
            return

    return digitos_em_zero


def converter_para_base(num_decimal, base):
    # num_decimal: inteiro -> representa o número decimal
    # base: inteiro -> representa a base para o número ser convertido
    # retorna uma lista com os dígitos na base exigida com a ordem invertida

    digitos = []
    while (num_decimal > 0):
        resto = num_decimal % base
        digitos.append(resto)
        num_decimal = num_decimal // base

    return digitos[::-1]
