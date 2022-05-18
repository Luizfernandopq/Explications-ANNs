def codify_network_tjeng(mdl, layers, input_variables, intermediate_variables, decision_variables, output_variables):
    # retorna ->        Bounds do modelo                   Variáveis de análise
    #                    list[min, max], [ub_menor_ou_igual_zero, lb_maior_ou_igual_zero, ub_lb_padrao]

    # Variáveis de análise:
    ub_y_less0 = 0
    lb_y_more0 = 0
    ub_lb_padrao = 0

    # Variáveis do modelo:
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()

        x = input_variables if i == 0 else intermediate_variables[i-1]

        if i != (len(layers)-1):
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            mdl.maximize(A[j, :] @ x + b[j])
            mdl.solve()
            ub = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if ub <= 0 and i != len(layers) - 1:
                mdl.add_constraint(y[j] == 0, ctname=f'c_{i}_{j}')

                # var de análise:
                ub_y_less0 += 1

                continue

            mdl.minimize(A[j, :] @ x + b[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')

                # var de análise:
                lb_y_more0 += 1
                continue

            if i != len(layers) - 1:
                mdl.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                mdl.add_constraint(y[j] >= A[j, :] @ x + b[j])
                mdl.add_constraint(y[j] <= ub * a[j])

                # var de análise:
                ub_lb_padrao += 1

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j])
                output_bounds.append([lb, ub])

    return output_bounds, [ub_y_less0, lb_y_more0, ub_lb_padrao]
