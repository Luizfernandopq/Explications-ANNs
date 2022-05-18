def codify_network_fischetti(mdl, layers, input_variables, auxiliary_variables, intermediate_variables,
                             decision_variables, output_variables):
    # retorna ->        Bounds do modelo                   Variáveis de análise
    #                    list[min, max]], [ub_y_menor_ou_igual_zero, ub_s_menor_ou_igual_zero, ub_y_ub_s_padrao]

    # Variáveis de análise:
    ub_y_less0 = 0
    ub_s_less0 = 0
    ub_y_ub_s_padrao = 0

    # Variáveis do modelo:
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()

        x = input_variables if i == 0 else intermediate_variables[i - 1]

        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            if i != len(layers) - 1:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j] - s[j], ctname=f'c_{i}_{j}')
                mdl.add_indicator(a[j], y[j] <= 0, 1)
                mdl.add_indicator(a[j], s[j] <= 0, 0)

                mdl.maximize(y[j])
                mdl.solve()
                ub_y = mdl.solution.get_objective_value()
                mdl.remove_objective()

                if ub_y <= 0:
                    mdl.remove_constraint(ct_arg=f'c_{i}_{j}')
                    mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')

                    # var de análise:
                    ub_y_less0 += 1

                    continue

                mdl.maximize(s[j])
                mdl.solve()
                ub_s = mdl.solution.get_objective_value()
                mdl.remove_objective()

                if ub_s <= 0:
                    mdl.remove_constraint(ct_arg=f'c_{i}_{j}')
                    mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')

                    # var de análise:
                    ub_s_less0 += 1

                    continue

                y[j].set_ub(ub_y)
                s[j].set_ub(ub_s)

                # var de análise:
                ub_y_ub_s_padrao += 1

            else:

                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')

                mdl.maximize(y[j])
                mdl.solve()
                ub = mdl.solution.get_objective_value()
                mdl.remove_objective()

                mdl.minimize(y[j])
                mdl.solve()
                lb = mdl.solution.get_objective_value()
                mdl.remove_objective()

                y[j].set_ub(ub)
                y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return output_bounds, [ub_y_less0, ub_s_less0, ub_y_ub_s_padrao]
