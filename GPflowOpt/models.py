from GPflow.param import Parameterized, AutoFlow, Param
from GPflow.model import GPModel
from GPflow.likelihoods import Gaussian
import GPflow
import tensorflow as tf

float_type = GPflow.settings.dtypes.float_type


class MGP(Parameterized):
    def __init__(self, obj):
        assert isinstance(obj, GPModel), "Class has to be a GP model"
        assert isinstance(obj.likelihood, Gaussian), "Likelihood has to be Gaussian"
        assert obj.Y.shape[1] == 1, "Only one dimensional functions are allowed"
        self.wrapped = obj
        self.cov_chol = None
        super(MGP, self).__init__()

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

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.build_predict(Xnew)

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
        L = self.cov_chol
        Dfmean = tf.stack(tf.gradients(fmean, c))
        Dfvar = tf.stack(tf.gradients(fvar, c))

        tmp1 = tf.matrix_triangular_solve(L, Dfmean)
        tmp2 = tf.matrix_triangular_solve(L, Dfvar)
        return fmean, 4 / 3 * fvar + tf.reduce_sum(tf.square(tmp1)) + 1 / 3 / (fvar+1E-3) * tf.reduce_sum(tf.square(tmp2))

    @AutoFlow()
    def _variance_cholesky(self):
        c = []
        self._unwrap(self.wrapped, c)
        h = -self._compute_hessian(self.build_likelihood(), c)
        diag = tf.expand_dims(tf.matrix_diag_part(h), -1)
        h = 1/diag*h/tf.transpose(diag) +tf.eye(len(c), dtype=float_type)*1E-3
        L = diag*tf.cholesky(h)
        return L

    def optimize(self, **kwargs):
        self.wrapped.optimize(**kwargs)
        self.cov_chol = Param(self._variance_cholesky())
