from gpflow.base import PriorOn


def randomize(model):
    for var in model.parameters:
        if hasattr(var, "prior") and var.prior is not None:
            if var.prior_on == PriorOn.CONSTRAINED:
                var.assign(var.prior.sample(var.shape))
            elif var.prior_on == PriorOn.UNCONSTRAINED:
                var.variables[0].assign(var.prior.sample(var.shape))
