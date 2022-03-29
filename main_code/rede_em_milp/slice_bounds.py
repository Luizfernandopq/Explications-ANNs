import numpy as np


def slice_bounds(bounds_input, num_de_sets):
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

    return combine_sliced_bounds(lista_de_bounds_input, len_bounds_input, num_de_sets)


def combine_slice_bounds_duplo(slices, num_de_variaveis):
    # slices: lista com pares de valores -> representa os limites
    # superior e inferior de cada variável fatiada

    sliced_bounds_input = []
    num_de_arranjos = 2 ** num_de_variaveis
    # num_de_arranjos: é um inteiro encontrado
    # a partir do número de fatias elevedado ao número de variáveis

    for i in range(num_de_arranjos):
        sliced_aux = []
        for j in range(num_de_variaveis):
            indice = int(i / 2 ** j)
            sliced_aux.append(slices[j][indice % 2])
        sliced_bounds_input.append(sliced_aux)

    return np.array(sliced_bounds_input), num_de_arranjos


def combine_sliced_bounds(slices, num_de_variaveis, num_de_slices):
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
            variavel = num_de_variaveis - (j+1)
            indice = int(i / num_de_slices ** variavel)
            sliced_aux.append(slices[j][indice % num_de_slices])
        sliced_bounds_input.append(sliced_aux)

    return np.array(sliced_bounds_input), num_de_arranjos


'''
    Abaixo há funções para auxiliar na lógica geral das funções acima 
    (Estas funções não estão sendo utilizadas e são obsoletas)
'''


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