
def codify_network_tjeng(mdl, layers, input_variables, intermediate_variables, decision_variables, output_variables):
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
                continue

            mdl.minimize(A[j, :] @ x + b[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f'c_{i}_{j}')
                continue

            if i != len(layers) - 1:
                mdl.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                mdl.add_constraint(y[j] >= A[j, :] @ x + b[j])
                mdl.add_constraint(y[j] <= ub * a[j])

                #modelo_em_milp.maximize(y[j])
                #modelo_em_milp.solve()
                #ub_y = modelo_em_milp.solution.get_objective_value()
                #modelo_em_milp.remove_objective()
                #y[j].set_ub(ub_y)

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j])
                #y[j].set_ub(ub)
                #y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mdl, output_bounds
