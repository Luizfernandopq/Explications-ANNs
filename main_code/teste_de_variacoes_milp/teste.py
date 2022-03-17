import numpy as np
import tensorflow as tf
from time import time
from statistics import mean, stdev
import pandas as pd

from main_code.rede_em_milp import main_milp as mm


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


def main():
    METODO_TJENG = False
    METODO_FISCHETTI = True

    datasets = [{'dir_path': 'australian', 'n_classes': 2},
                {'dir_path': 'auto', 'n_classes': 5},
                {'dir_path': 'backache', 'n_classes': 2},
                {'dir_path': 'breast-cancer', 'n_classes': 2},
                {'dir_path': 'cleve', 'n_classes': 2},
                {'dir_path': 'cleveland', 'n_classes': 5},
                {'dir_path': 'glass', 'n_classes': 5},
                {'dir_path': 'glass2', 'n_classes': 2},
                {'dir_path': 'heart-statlog', 'n_classes': 2},
                {'dir_path': 'hepatitis', 'n_classes': 2},
                {'dir_path': 'spect', 'n_classes': 2},
                {'dir_path': 'voting', 'n_classes': 2}
                ]

    configurations = [{'method': METODO_FISCHETTI},
                      {'method': METODO_TJENG}]

    df = {'fischetti': {'size': [], 'milp_time': [], 'build_time': []},
          'tjeng': {'size': [], 'milp_time': [], 'build_time': []}}

    slices = [1]

    for dataset in datasets:
        dir_path = dataset['dir_path']
        n_classes = dataset['n_classes']

        for config in configurations:
            print(dataset, config)

            method = config['method']

            data_test = pd.read_csv(f'../../datasets/{dir_path}/test.csv')
            data_train = pd.read_csv(f'../../datasets/{dir_path}/train.csv')

            data = data_train.append(data_test)

            model_path = f'../../datasets/{dir_path}/model_1layers_40neurons_{dir_path}.h5'
            model = tf.keras.models.load_model(model_path)

            codify_network_time = []
            for slice in slices:
                start = time()
                # mdl, output_bounds = mm.codify_network(model, data, method, slice)

                lista_de_modelos_em_milp, lista_de_bounds = mm.codify_network(model, data, method, slice)

                codify_network_time.append(time() - start)
                print(codify_network_time[-1])

            time_list = []
            len_list = []
            data = data.to_numpy()
            print(data)
            for i in range(data.shape[0]):
                # if i % 50 == 0:
                print(f'Unidade: {i}')
                network_input = data[i, :-1]

                network_input = tf.reshape(tf.constant(network_input), (1, -1))
                network_output = model.predict(tf.constant(network_input))[0]
                network_output = tf.argmax(network_output)

                aux_lenlist = []
                for j in range(len(lista_de_modelos_em_milp)):
                    print(f'slice (max = {slices[j]}^{data.shape[1]}): {j+1}')
                    mdl_aux = lista_de_modelos_em_milp[j].clone()
                    output_bounds = lista_de_bounds[j]
                    start = time()

                    explanation = get_miminal_explanation(mdl_aux, network_input, network_output,
                                                          n_classes=n_classes, method=method,
                                                          output_bounds=output_bounds)

                    print(explanation)
                    time_list.append(time() - start)
                    aux_lenlist.append(len(explanation))

                len_list.append(aux_lenlist)

            print(f'\n{len_list}')

            df[method]['size'].extend([min(len_list), f'{mean(len_list)} +- {stdev(len_list)}', max(len_list)])
            df[method]['milp_time'].extend([min(time_list), f'{mean(time_list)} +- {stdev(time_list)}', max(time_list)])
            df[method]['build_time'].extend([min(codify_network_time),
                                             f'{mean(codify_network_time)} +- {stdev(codify_network_time)}',
                                             max(codify_network_time)])

            print(
                f'Explication sizes:\nm: {min(len_list)}\na: {mean(len_list)} +- {stdev(len_list)}\nM: {max(len_list)}')
            print(f'Time:\nm: {min(time_list)}\na: {mean(time_list)} +- {stdev(time_list)}\nM: {max(time_list)}')
            print(
                f'Build Time:\nm: {min(codify_network_time)}\na: {mean(codify_network_time)} +- {stdev(codify_network_time)}\nM: {max(codify_network_time)}')

    df = {'fischetti_size': df[METODO_FISCHETTI]['size'],
          'fischetti_time': df[METODO_FISCHETTI]['milp_time'],
          'fischetti_build_time': df[METODO_FISCHETTI]['build_time'],
          'tjeng_size': df[METODO_TJENG]['size'],
          'tjeng_time': df[METODO_TJENG]['milp_time'],
          'tjeng_build_time': df[METODO_TJENG]['build_time']}

    index_label = []
    for dataset in datasets:
        index_label.extend([f"{dataset['dir_path']}_m", f"{dataset['dir_path']}_a", f"{dataset['dir_path']}_M"])
    df = pd.DataFrame(data=df, index=index_label)
    df.to_csv('results2.csv')


if __name__ == '__main__':
    # cProfile.run('main()', sort='time')
    main()
