from .design import Design
from .domain import Domain
from .factory import ModelFactory
import numpy as np
from gpflow.models import BayesianModel
from typing import Callable, List, Optional
from .typing import LabeledData


class Experiment:

    @staticmethod
    def create(initial_design: Design, fx: Callable) -> 'Experiment':
        data = Experiment(initial_design.domain)
        data.points = initial_design.generate()
        data.evaluations = fx(data.points)
        return data

    def __init__(self, domain: Domain) -> None:
        super().__init__()
        self.domain = domain
        self.points = None
        self.evaluations = None

    def training_data(self, output_idx: Optional[List[int]] = None) -> LabeledData:
        output_idx = output_idx or list(range(self.evaluations.shape[1]))
        return self.points, self.evaluations[:, np.atleast_1d(output_idx)]

    def models(self, factory: ModelFactory) -> List[BayesianModel]:
        # permit a dict of factories?
        # todo: we'll be changing the way outputs are linked to acquisitions
        models = []
        for out_idx in range(self.evaluations.shape[1]):
            data = self.training_data(out_idx)
            m = factory.create_model(data)
            models.append(m)
        return models

    def add(self, data: LabeledData) -> 'Experiment':
        x, y = data
        self.points = np.vstack((self.points, np.atleast_2d(x)))
        self.evaluations = np.vstack((self.evaluations, np.atleast_2d(x)))
        return self

