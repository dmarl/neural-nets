# neural-nets

A from-scratch implementation of a vanilla neural network with no reliance upon existing ML libraries

uses mini-batch stochastic gradient descent with momentum and gives a choice of activation/cost functions

trained on the MNIST digits dataset, data available on github

instance of a neural network might look like:

import nnets as nn
Network = nn.nnet([10, 100, 100, 10], cost=nn.crossentropy)
Network.fit(training_data, eta=.5, batch_size=10, epochs=100, lmbda=1.0, mu=0.5, test=None, graph=True)
Network.score(test_data),

where:
- eta is the learning rate
- batch_size is the size of minibatches (1 for on-line learning)
- epochs is the number of training epochs
- lmbdda is the regularisation parameter
- mu is the friction parameter for momentum (0 for learning without using momentum)
- test is (optional) test data for outputting the learning rate
- graph = True plots the classification success rate by epoch on training (and test if selected) data
