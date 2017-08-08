Acquisition functions
========================

The GPflowOpt package currently supports a limited number of popular acquisition functions. These are
summarized in the table below. Detailed description for each can be found below.

.. automodule:: GPflowOpt.acquisition

+----------------------------------------------------------+-----------+-------------+-----------+
|  Method                                                  | Objective |  Constraint | # Outputs |
+==========================================================+===========+=============+===========+
| :class:`GPflowOpt.acquisition.ExpectedImprovement`       |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`GPflowOpt.acquisition.ProbabilityOfFeasibility`  |           |      ✔      |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`GPflowOpt.acquisition.ProbabilityOfImprovement`  |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`GPflowOpt.acquisition.LowerConfidenceBound`      |     ✔     |             |     1     |
+----------------------------------------------------------+-----------+-------------+-----------+
| :class:`GPflowOpt.acquisition.HVProbabilityOfImprovement`|     ✔     |             |    > 1    |
+----------------------------------------------------------+-----------+-------------+-----------+

Single-objective
----------------

Expected Improvement
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPflowOpt.acquisition.ExpectedImprovement
   :members:
   :special-members:

Probability of Feasibility
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPflowOpt.acquisition.ProbabilityOfFeasibility
   :members:
   :special-members:

Probability of Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPflowOpt.acquisition.ProbabilityOfImprovement
   :members:
   :special-members:

Lower Confidence Bound
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPflowOpt.acquisition.LowerConfidenceBound
   :members:
   :special-members:

Multi-objective
----------------

Hypervolume-based Probability of Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPflowOpt.acquisition.HVProbabilityOfImprovement
   :members:
   :special-members:

Pareto module
^^^^^^^^^^^^^

.. automodule:: GPflowOpt.pareto
   :members:
.. automethod:: GPflowOpt.pareto.Pareto.hypervolume
