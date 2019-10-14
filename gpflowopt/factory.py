from abc import ABC, abstractmethod
from gpflow.models import BayesianModel, GPR
from gpflow.kernels import RBF
from gpflow.optimizers import Scipy
from .typing import LabeledData


class ModelFactory(ABC):

    @abstractmethod
    def create_model(self, data: LabeledData) -> BayesianModel:
        pass

    def update_model(self, model: BayesianModel, data: LabeledData) -> BayesianModel:
        return self.create_model(data)


class BasicGPRFactory(ModelFactory):

    def __init__(self, ard=None):
        super().__init__()
        self.ard = ard;

    def _optimize_gpr(self, model: GPR) -> None:
        opt = Scipy()
        opt.minimize(model.neg_log_marginal_likelihood, model.trainable_variables, options=dict(maxiter=100))

    def create_model(self, data: LabeledData) -> GPR:
        # TODO @nknudde: make model more involved (priors, assigning initial values, ...)
        m = GPR(data, RBF(ard=self.ard))
        self._optimize_gpr(m)
        return m

    def update_model(self, model: GPR, data: LabeledData) -> GPR:
        model.data = data
        self._optimize_gpr(model)
        return model