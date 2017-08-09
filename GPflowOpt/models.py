from GPflow.param import Parameterized, AutoFlow, Param
from GPflow.model import Model, GPModel
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
    num_feat = tf.shape(X)[0]

    def body(old_grads, row):
        g = tf.expand_dims(tf.gradients(Y[row], X)[0], axis=0)
        new_grads = tf.concat([old_grads, g], axis=0)
        return new_grads, row + 1

    def cond(_, row):
        return tf.less(row, num_rows)

    shape_invariants = [tf.TensorShape([None, None]), tf.TensorShape([])]
    grads, _ = tf.while_loop(cond, body, [tf.zeros([0, num_feat], float_type), tf.constant(0)],
                             shape_invariants=shape_invariants)

    return grads


class MGP(Model):
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
        self.wrapped = obj
        super(MGP, self).__init__(name=obj.name + "_MGP")

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

    def build_predict(self, fmean, fvar, theta):
        h = tf.hessians(self.build_likelihood() + self.build_prior(), theta)[0]
        L = tf.cholesky(-h)

        N = tf.shape(fmean)[0]
        D = tf.shape(fmean)[1]

        fmeanf = tf.reshape(fmean, [N * D, 1])      # N*D x 1
        fvarf = tf.reshape(fvar, [N * D, 1])        # N*D x 1

        Dfmean = rowwise_gradients(fmeanf, theta)   # N*D x k
        Dfvar = rowwise_gradients(fvarf, theta)     # N*D x k

        tmp1 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfmean)))    # N*D x k
        tmp2 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfvar)))     # N*D x k
        return fmean, 4 / 3 * fvar + tf.reshape(tf.reduce_sum(tf.square(tmp1), axis=1), [N, D]) \
               + 1 / 3 / (fvar + 1E-3) * tf.reshape(tf.reduce_sum(tf.square(tmp2), axis=1), [N, D])

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        theta = self._predict_f_AF_storage['free_vars']
        fmean, fvar = self.wrapped.build_predict(Xnew)
        return self.build_predict(fmean, fvar, theta)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        theta = self._predict_y_AF_storage['free_vars']
        pred_f_mean, pred_f_var = self.wrapped.build_predict(Xnew)
        fmean, fvar = self.wrapped.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        return self.build_predict(fmean, fvar, theta)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        theta = self._predict_density_AF_storage['free_vars']
        pred_f_mean, pred_f_var = self.wrapped.build_predict(Xnew)
        pred_f_mean, pred_f_var = self.build_predict(pred_f_mean, pred_f_var, theta)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
