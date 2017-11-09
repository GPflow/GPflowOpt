Acquisition functions
========================

The gpflowopt package currently supports a limited number of popular acquisition functions. These are
summarized in the table below. Detailed description for each can be found below.

.. automodule:: gpflowopt.acquisition

+----------------------------------------------------------+-----------+-------------+-----------+
|  Method                                                  | Objective |  Constraint | # Outputs |
+==========================================================+===========+=============+===========+
| :class:`gpflowopt.acquisition.ExpectedImprovement`       |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`gpflowopt.acquisition.ProbabilityOfFeasibility`  |           |      ✔      |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`gpflowopt.acquisition.ProbabilityOfImprovement`  |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`gpflowopt.acquisition.LowerConfidenceBound`      |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`gpflowopt.acquisition.MinValueEntropySearch`     |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`gpflowopt.acquisition.HVProbabilityOfImprovement`|     ✔     |             |    > 1    |
+----------------------------------------------------------+-----------+-------------+-----------+

Single-objective
----------------

Expected Improvement
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.ExpectedImprovement
   :members:
   :special-members:

Probability of Feasibility
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.ProbabilityOfFeasibility
   :members:
   :special-members:

Probability of Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.ProbabilityOfImprovement
   :members:
   :special-members:

Lower Confidence Bound
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.LowerConfidenceBound
   :members:
   :special-members:

Min-Value Entropy Search
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.MinValueEntropySearch
   :members:
   :special-members:


Multi-objective
----------------

Hypervolume-based Probability of Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpflowopt.acquisition.HVProbabilityOfImprovement
   :members:
   :special-members:

Pareto module
^^^^^^^^^^^^^

.. automodule:: gpflowopt.pareto
   :members:
.. automethod:: gpflowopt.pareto.Pareto.hypervolume
