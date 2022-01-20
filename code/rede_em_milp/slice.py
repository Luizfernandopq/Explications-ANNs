def slice_bounds(bounds_input, num_de_sets):
    lista_de_bounds_input = []
    for linha in bounds_input:

        amplitude_relativa = (linha[1] - linha[0]) / num_de_sets
        slices = []
        for i in range(num_de_sets):

            slices.append([linha[0] + amplitude_relativa * i, linha[0] + amplitude_relativa * (i + 1)])
        lista_de_bounds_input.append(slices)

    return combine_sliced_bounds(lista_de_bounds_input, num_de_sets)


def combine_sliced_bounds(slices, num_de_slices):
    # slices: lista com pares de valores -> representa os limites
    # superior e inferior de cada variável fatiada
    # num_de_slices: inteiro -> representa o número de fatias da variável slices

    print(f"def combine_slice \n{slices}\n")

    sliced_bounds_input = []
    num_de_variaveis = len(slices)
    num_de_arranjos = num_de_slices ** num_de_variaveis
    # num_de_arranjos: é um inteiro encontrado
    # a partir do número de fatias elevedado ao número de variáveis

    digitos = converter_para_maximo_da_base(num_de_variaveis, num_de_slices)

    for i in range(num_de_arranjos):
        sliced_aux = []
        for j in range(num_de_variaveis):
            endereco_da_variavel = digitos[j]
            sliced_aux.append(slices[j][endereco_da_variavel])

        digitos = proximo(digitos, num_de_slices)
        sliced_bounds_input.append(sliced_aux)

    return sliced_bounds_input


'''
    Abaixo há funções para auxiliar na lógica geral das funções acima
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
