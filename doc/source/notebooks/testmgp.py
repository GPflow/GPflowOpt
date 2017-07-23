from GPflow.gpr import GPR
import numpy as np
from GPflow.kernels import RBF, White
from GPflow.priors import Gamma, LogNormal
from GPflowOpt.models import MGP
from GPflow.transforms import Log1pe
import matplotlib.pyplot as plt


np.random.seed(3)
x = np.random.rand(8, 1) * 8-4
y = np.cos(x) +np.random.randn(8,1)*0.1

m = MGP(GPR(x, y, RBF(1, lengthscales=1, variance=1)))
m.kern.lengthscales.prior = Gamma(3, 1/3)
m.kern.variance.prior = Gamma(3, 1/3)
m.likelihood.variance.prior = LogNormal(0, 1)
m.optimize(display=True)
print(m)
print('Trained')

X = np.array(np.linspace(-4, 4, 100)[:, None])
fms, fvs = m.predict_f(X)
print(fms, fvs)
print('Evaluated')

plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.plot(X.flatten(), fms.flatten())
plt.fill_between(X.flatten(), fms.flatten() - 2 * np.sqrt(fvs.flatten()), fms.flatten() + 2 * np.sqrt(fvs.flatten()),
                 alpha=0.2)
plt.title('MGP - GPR')
plt.subplot(1, 2, 2)
plt.scatter(x, y)
fm, fv = m.wrapped.predict_f(X)
plt.plot(X.flatten(), fm.flatten())
plt.fill_between(X.flatten(), fm.flatten() - 2 * np.sqrt(fv.flatten()), fm.flatten() + 2 * np.sqrt(fv.flatten()),
                 alpha=0.2)
plt.title('GPR')
plt.show()