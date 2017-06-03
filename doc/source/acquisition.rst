Acquisition functions
========================

The GPflowOpt package currently supports a limited number implementations of popular acquisition functions. These are
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


Expected Improvement
--------------------

.. autoclass:: GPflowOpt.acquisition.ExpectedImprovement
   :members:
   :special-members:

Probability of Feasibility
--------------------------

.. autoclass:: GPflowOpt.acquisition.ProbabilityOfFeasibility
   :members:
   :special-members:

Probability of Improvement
--------------------------

.. autoclass:: GPflowOpt.acquisition.ProbabilityOfImprovement
   :members:
   :special-members:

Lower Confidence Bound
--------------------------

.. autoclass:: GPflowOpt.acquisition.LowerConfidenceBound
   :members:
   :special-members: