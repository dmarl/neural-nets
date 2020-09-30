import numpy as np
import random
import matplotlib.pyplot as plt

import mnist_loader as mn
trda, vada, teda = mn.load_data_wrapper()
xs = list(trda)
ys=list(teda)

def dropout(X, p):
    mask = np.random.uniform(0.0, 1.0, size = X.shape)
    return X*mask/(1-p)

class sig(object):
    @staticmethod
    def fn(z):
        return 1.0/(1.0+np.exp(-z))
        
    def fn_prime(z):
        return np.exp(-z)/(1+np.exp(-z))**2

class tanh(object):
    def fn(z):
        return (np.exp(2*z)-1)/(np.exp(2*z)+1)
        
    def fn_prime(z):
        return 4.0/(np.exp(-2*z)+2+np.exp(2*z))

class softplus(object):
    def fn(z):
        return np.log(1+np.exp(z))
        
    def fn_prime(z):
        return 1.0/(1.0+np.exp(-z))

def softmax(xs):
    ys = [np.exp(x) for x in xs]
    y = sum(ys)
    zs = np.zeros(xs.shape)
    for i in range(len(xs)):
        zs[i] = ys[i]/y
    return zs

class quadcost(object):
    @staticmethod
    def fn(a, y):
        return .5*np.linalg.norm(a-y)**2
        
    def delta(z, a, y):
        return (a-y)*act_fn.fn_prime(z)
    
class crossentropy(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        
    def delta(z, a, y):
        return (a-y)

class loglikelihood(object):
    @staticmethod
    def fn(a, y):
        return -np.log(a[np.argmax(y)])
    
    def delta(z, a, y):
        return (a-y)

class nnet(object):
    def __init__(self, layers, act_fn=sig, cost=quadcost, p_drop=0, decay_rate=1):
        self.depth=len(layers)
        self.layers=layers
        # random initial assignment of weights and biases
        self.weights=[np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases=[np.random.randn(x, 1) for x in layers[1:]]
        self.cost=cost
        self.act_fn = act_fn
        self.p_drop = p_drop
        self.decay_rate=decay_rate
        
    def feed_forward(self, a):
        # compute network output
        for bias, weight in zip(self.biases, self.weights):
            a = self.act_fn.fn(np.dot(weight, a)+bias)
        return a
    
    def fit(self, training, eta=.5, batch_size=10, epochs=50, lmbda=0.0, mu=0.0, test=None, graph=True, w_max=None):
        # stochastic gradient descent with momentum and regularization in
        # in epochs on minibatches of size batch_size on tuples of training
        #  data (training) with learning rate eta, regularity parameter
        # lmbda and friction mu
        lc = []
        tc=[]
        n=len(training)
        for i in range(epochs):
            print("Epoch {}".format(i+1))
            random.shuffle(training)
            batches = [training[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.batch_update(batch, eta, lmbda, mu, len(training), w_max)
            training_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in training]
            lc.append(sum(int(x==y) for (x, y) in training_results)/len(training_results))
            if test:
                test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test]
                tc.append(sum(int(x==y) for (x, y) in test_results)/len(test_results))
            eta *= self.decay_rate
        xs = [i for i in range(epochs)]
        if graph:
            plt.plot(xs, lc, color='green')
            if test:
                plt.plot(xs, tc, color='red')
            plt.show()
            
    def batch_update(self, batch, eta, mu, lmbda, n, w_max):
        # updates weights and biases in batches
        
        vels = [np.zeros(weight.shape) for weight in self.weights]
        del_b = [np.zeros(bias.shape)  for bias in self.biases]
        del_w = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            del_b = [db + dtb for db, dtb in zip(del_b, delta_b)]
            del_w = [dw + dtw for dw, dtw in zip(del_w, delta_w)]
        self.biases = [bias - db*eta/len(batch) for bias, db in zip(self.biases, del_b)]
        self.weights = [weight*(1-lmbda*eta/n)-dw*eta/len(batch) + mu*vel for weight, dw, vel in zip(self.weights, del_w, vels)]  # final batch not guaranteed to be of size batch_size...
        if w_max:
            self.weights = [weight * w_max / np.linalg.norm(weight) if np.linalg.norm(weight) > w_max else weight for weight in self.weights]
        vels = [-dw*eta/len(batch) for dw in del_w]

    def backprop(self, x, y):
        # computes partial derivatives of cost function (self.cost)
        # wrto weights and biases
        
        del_b = [np.zeros(bias.shape) for bias in self.biases]
        del_w = [np.zeros(weight.shape) for weight in self.weights]
        
        # process outputs
        act = x
        acts  = [x]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, act) + bias
            zs.append(z)
            act = self.act_fn.fn(z)
            if self.p_drop:
                act = dropout(act, self.p_drop)
            acts.append(act)
            
        
        delta = (self.cost).delta(zs[-1], acts[-1], y)
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, acts[-2].transpose())
        
        # backpropagate
        for i in range(2, self.depth):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.act_fn.fn_prime(zs[-i])
            del_b[-i] = delta
            del_w[-i] = np.dot(delta, acts[-i-1].transpose())
        return (del_b, del_w)
    
    def score(self, test):
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test]
        return sum(int(x==y) for (x, y) in results)/len(results)

# convolutional nets !
# adding capability for one or more convolutional (and pooling)
# layers before fully-connected layers and output
