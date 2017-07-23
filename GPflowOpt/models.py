from GPflow.param import Parameterized, AutoFlow, Param
from GPflow.model import GPModel
from GPflow.likelihoods import Gaussian
import GPflow
import tensorflow as tf

float_type = GPflow.settings.dtypes.float_type


def rowwise_gradients(Y, X):
    """
    For a 2D Tensor Y, compute the derivatiave of each columns w.r.t  a 2D tensor X.

    This is done with while_loop, because of a known incompatibility between map_fn and gradients.
    """
    num_rows = tf.shape(Y)[0]

    def body(old_grads, row):
        g = tf.stack(tf.gradients(Y[row], X), axis=1)
        new_grads = tf.concat([old_grads, g], axis=0)
        return new_grads, row + 1

    def cond(_, row):
        return tf.less(row, num_rows)

    shape_invariants = [tf.TensorShape([None, len(X)]), tf.TensorShape([])]
    grads, _ = tf.while_loop(cond, body, [tf.zeros([0, len(X)], float_type), tf.constant(0)],
                             shape_invariants=shape_invariants)

    return grads


class MGP(GPModel):
    """
    Marginalisation of the hyperparameters during evaluation time using a Laplace Approximation
    Key reference:

    ::

       @article{garnett2013active,
          title={Active learning of linear embeddings for Gaussian processes},
          author={Garnett, Roman and Osborne, Michael A and Hennig, Philipp},
          journal={arXiv preprint arXiv:1310.6740},
          year={2013}
        }
    """

    def __init__(self, obj):
        assert isinstance(obj, GPModel), "Class has to be a GP model"
        assert isinstance(obj.likelihood, Gaussian), "Likelihood has to be Gaussian"
        assert obj.Y.shape[1] == 1, "Only one dimensional functions are allowed"
        self.wrapped = obj
        self.cov_chol = None
        super(MGP, self).__init__(None, None, None, None, 1, name=obj.name + "_MGP")
        del self.kern
        del self.mean_function
        del self.likelihood
        del self.X
        del self.Y

    def __getattr__(self, item):
        """
        If an attribute is not found in this class, it is searched in the wrapped model
        """
        return self.wrapped.__getattribute__(item)

    def __setattr__(self, key, value):
        """
        If setting :attr:`wrapped` attribute, point parent to this object (the datascaler)
        """
        if key is 'wrapped':
            object.__setattr__(self, key, value)
            value.__setattr__('_parent', self)
            return

        super(MGP, self).__setattr__(key, value)

    def _unwrap(self, p, c):
        """
        Unwrap all the parameters captured in the model
        """
        if isinstance(p, Param):
            c.append(p._tf_array)
        elif isinstance(p, Parameterized):
            for p2 in p.sorted_params:
                self._unwrap(p2, c)

    def _compute_hessian(self, x, y):
        mat = []
        for v1 in y:
            temp = []
            for v2 in y:
                # computing derivative twice, first w.r.t v2 and then w.r.t v1
                temp.append(tf.gradients(tf.gradients(x, v2)[0], v1)[0])
            temp = [tf.constant(0, dtype=float_type) if t is None else t for t in
                    temp]
            temp = tf.stack(temp)
            mat.append(temp)
        mat = tf.squeeze(tf.stack(mat))
        return mat

    def build_predict(self, Xnew, full_cov=False):
        fmean, fvar = self.wrapped.build_predict(Xnew=Xnew, full_cov=full_cov)
        c = []
        self._unwrap(self.wrapped, c)
        C = tf.expand_dims(tf.stack(c, axis=0), 1)

        Dfmean = rowwise_gradients(fmean, c)
        Dfvar = rowwise_gradients(fvar, c)

        h = -self._compute_hessian(self.build_likelihood() + self.build_prior(), c)
        L = tf.cholesky(h)

        tmp1 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfmean)))
        tmp2 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfvar)))
        return fmean, 4 / 3 * fvar + tf.expand_dims(tf.reduce_sum(tf.square(tmp1), axis=1), 1) \
               + 1 / 3 / (fvar + 1E-3) * tf.expand_dims(tf.reduce_sum(tf.square(tmp2), axis=1), 1)
