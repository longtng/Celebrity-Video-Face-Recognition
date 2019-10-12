import numpy as np
from sklearn.datasets import load_boston
X  = load_boston(return_X_y=True)
X_train =  X[0]
y_train = X[1]
#@print(X_train)
print(X_train.shape)
print(np.amax(X_train))
print(np.amin(X_train))

### Use MondrianForests for variance estimation
from skgarden import MondrianForestRegressor
mfr = MondrianForestRegressor()
mfr.fit(X_train, y_train)
y_mean, y_std = mfr.predict(X_train, return_std=True)
print(y_mean)
#print(y_std)

### Use QuantileForests for quantile estimation
#from skgarden import RandomForestQuantileRegressor
#rfqr = RandomForestQuantileRegressor(random_state=0)
#rfqr.fit(X, y)
#y_mean = rfqr.predict(X)
#y_median = rfqr.predict(X, 50)
