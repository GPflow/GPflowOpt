from GPflow.gpr import GPR
import numpy as np
from GPflow.kernels import RBF, White
from GPflowOpt.models import MGP
import matplotlib.pyplot as plt

x = np.random.randn(4, 1)*2

y = np.sin(x)

m = MGP(GPR(x, y, RBF(1, lengthscales=1, variance=1) + White(1)))
m.optimize()
print(m)
print('Trained')

X = np.array(np.linspace(-3, 3, 100)[:, None])
fm = []
fv = []
for xn in X:
    fms, fvs = m.predict_f(xn.reshape(-1, 1))
    fm.append(fms)
    fv.append(fvs)
fm = np.stack(fm, axis=0)
fv = np.stack(fv, axis=0)
print('Evaluated')

plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.plot(X.flatten(), fm.flatten())
plt.fill_between(X.flatten(), fm.flatten() - 2 * np.sqrt(fv.flatten()), fm.flatten() + 2 * np.sqrt(fv.flatten()),
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
