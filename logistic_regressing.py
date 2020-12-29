import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def initialize_params(dims):
    W = np.zeros((dims, 1))
    b = 0
    return W, b

def logistic(X, y, W, b):
    num_train = X.shape[0]
    num_feature = X.shape[1]

    a = sigmoid(np.dot(X, W) + b)
    cost = -1/num_train * np.sum(y*np.log(a) + (1-y)*np.log(1-a))

    dW = np.dot(X.T, (a-y))/num_train
    db = np.sum(a-y)/num_train
    cost = np.squeeze(cost)

    return a, cost, dW, db

def logistic_train(X, y, learning_rate, epochs):
    # 初始化模型参数
    W, b = initialize_params(X.shape[1])
    cost_list = []

    # 迭代训练
    for i in range(epochs):
        # 计算当前次的模型计算结果、损失和参数梯度
        a, cost, dW, db = logistic(X, y, W, b)
        # 参数更新
        W = W -learning_rate * dW
        b = b -learning_rate * db

        # 记录损失
        if i % 100 == 0:
            cost_list.append(cost)
        # 打印训练过程中的损失
        if i % 100 == 0:
            print('epoch %d cost %f' % (i, cost))

    # 保存参数
    params = {
        'W': W,
        'b': b
    }
    # 保存梯度
    grads = {
        'dW': dW,
        'db': db
    }
    return cost_list, params, grads

def predict(X, params):
    y_prediction = sigmoid(np.dot(X, params['W']) + params['b'])
    for i in range(len(y_prediction)):
        if y_prediction[i] > 0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0
    return y_prediction

X,labels=make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2)
rng=np.random.RandomState(2)
X+=2*rng.uniform(size=X.shape)

unique_lables=set(labels)
colors=plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
for k, col in zip(unique_lables, colors):
    x_k=X[labels==k]
    plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
             markersize=14)
# plt.title('data by make_classification()')
# plt.show()

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)
cost_list, params, grads = logistic_train(X_train, y_train, 0.01, 1000)

y_prediction = predict(X_test, params)
print(y_prediction)


def accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            if y_test[i] == y_pred[j] and i == j:
                correct_count += 1

    accuracy_score = correct_count / len(y_test)
    return accuracy_score


def plot_logistic(X_train, y_train, params):
    n = X_train.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=32, c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    x = np.arange(-1.5, 3, 0.1)
    y = (-params['b'] - params['W'][0] * x) / params['W'][1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


plot_logistic(X_train, y_train, params)