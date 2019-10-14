from typing import Tuple
import tensorflow as tf

LabeledData = Tuple[tf.Tensor, tf.Tensor]
Prediction = Tuple[tf.Tensor, tf.Tensor]
