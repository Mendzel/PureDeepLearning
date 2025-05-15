from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from PureDeepLearning.main.activations import Linear, Sigmoid
from PureDeepLearning.main.base import NeuralNetwork, Trainer
from PureDeepLearning.main.layers import Dense
from PureDeepLearning.main.losses import MeanSquaredError
from PureDeepLearning.main.optimizers import SGD, SGDMomentum
from PureDeepLearning.utils.helpers import to_2d_np
from PureDeepLearning.utils.metrics import eval_regression_model

# Loading the data
housing = fetch_california_housing()
data = housing.data
target = housing.target
features = housing.feature_names

# Scaling the data
s = StandardScaler()
data = s.fit_transform(data)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

##### MODEL 1 #####
lr = NeuralNetwork(
    layers=[Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)
trainer = Trainer(lr, SGD(lr=0.01))
trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501)
eval_regression_model(lr, X_test, y_test)

##### MODEL 2 #####
dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                  activation=Sigmoid(),
                  dropout=0.9),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer = Trainer(dl, SGDMomentum(lr=0.01, momentum=0.85, final_lr=0.005, decay_type='linear'))
trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 100,
       eval_every = 10,
       seed=20190501)
eval_regression_model(dl, X_test, y_test)

##### MODEL 3 #####
trainer = Trainer(dl, SGD(lr=0.01))
trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501)
eval_regression_model(dl, X_test, y_test)