
def conv(m, n, i, j):
    (a, b) = n.shape
    return sum(m[i+k][j+l]*n[k][l] for k in range(a) for l in range(b))

def repack_im(im, dim):
    sqim = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            sqim[i][j] = im[dim*i+j]
    return sqim

def unravel(xs, l, m, n):
    tsr = [np.zeros((m, n)) for i in range(l)]
    for i in range(l):
        for j in range(m):
            for k in range(n):
                tsr[i][j][k] = xs[i*m*n+j*n+k]
    return tsr
    
class conv_nn(object):
    def __init__(self, feat_maps=3, dim_im=(28,28), dim_filt=(5,5), layers=[10], dim_pool=(2,2), act_fn = sig, cost=crossentropy):
        # convolutional neural net, receiving as input an image of dimension dim_im (rows x cols)
        # with one convolution layer (with feat_maps feature maps and filter of size dim_filter
        # (rows x cols)), pooling (with pool size dim_pool), and subsequent fully
        # connected layers specified by layers
        self.depth=len(layers)+1
        self.dim_pool = dim_pool
        self.feat_maps = feat_maps
        self.dim_im = dim_im
        self.dim_filt = dim_filt
        self.layers = [int(feat_maps*(dim_im[0]-dim_filt[0]+1)*(dim_im[1]-dim_filt[1]+1)/(dim_pool[0]*dim_pool[1]))]+layers
        self.conv_ws = [np.random.randn(dim_filt[0],dim_filt[1]) for i in range(feat_maps)]
        self.conv_bs = [np.random.randn() for i in range(feat_maps)]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.randn(x,1) for x in self.layers[1:]]
        self.cost = cost
        self.act_fn = act_fn
    
    def feed_forward(self, a, backprop=False):
        x0 = self.dim_im[0]-self.dim_filt[0]+1
        y0 = self.dim_im[1]-self.dim_filt[1]+1
        x1 = int((self.dim_im[0]-self.dim_filt[0]+1)/self.dim_pool[0])
        y1 = int((self.dim_im[1]-self.dim_filt[1]+1)/self.dim_pool[1])
        convs = [np.zeros((x0,y0)) for i in range(self.feat_maps)]
        pools = [np.zeros((x1, y1)) for i in range(self.feat_maps)]
        convz = [np.zeros((x0,y0)) for i in range(self.feat_maps)]
        poolz = [np.zeros((x1, y1)) for i in range(self.feat_maps)]
                
        zs = []
        acts = []
        
        for i in range(self.feat_maps):
            for j in range(x0):
                for k in range(y0):
                    convz[i][j][k] = conv(a, self.conv_ws[i], j, k)+self.conv_bs[i]
                    convs[i][j][k] = self.act_fn.fn(convz[i][j][k])
        zs.append(convz)
        acts.append(convs)
        for i in range(self.feat_maps):
            for j in range(x1):
                for k in range(y1):
                    poolz[i][j][k] = np.sqrt(sum([convz[i][j*self.dim_pool[0]+m][k*self.dim_pool[1]+n]**2 for m in range(self.dim_pool[0]) for n in range(self.dim_pool[1])]))
                    pools[i][j][k] = np.sqrt(sum([convs[i][j*self.dim_pool[0]+m][k*self.dim_pool[1]+n]**2 for m in range(self.dim_pool[0]) for n in range(self.dim_pool[1])]))
        postpoolz = np.array([val for i in range(self.feat_maps) for val in list(poolz[i].ravel())]).reshape(self.feat_maps*x1*y1, 1)
        postpool = np.array([val for i in range(self.feat_maps) for val in list(pools[i].ravel())]).reshape(self.feat_maps*x1*y1, 1)
        zs.append(postpoolz)
        acts.append(postpool)
        for weight, bias in zip(self.weights, self.biases):
            postpool = np.dot(weight, postpool) + bias
            zs.append(postpool)
            postpool=self.act_fn.fn(postpool)
            acts.append(postpool)
        if backprop:
            return acts, zs
        return acts[-1]

    def backprop(self, x, y):
        x0 = self.dim_im[0]-self.dim_filt[0]+1
        y0 = self.dim_im[1]-self.dim_filt[1]+1
        x1 = int((self.dim_im[0]-self.dim_filt[0]+1)/self.dim_pool[0])
        y1 = int((self.dim_im[1]-self.dim_filt[1]+1)/self.dim_pool[1])

        acts, zs = self.feed_forward(x, backprop=True)
        
        del_conv_b = [0 for bias in self.conv_bs]
        del_conv_w = [np.zeros(weight.shape) for weight in self.conv_ws]
        
        del_b = [np.zeros(bias.shape) for bias in self.biases]
        del_w = [np.zeros(weight.shape) for weight in self.weights]
        
        delta = (self.cost).delta(zs[-1], acts[-1], y)
        
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, acts[-2].transpose())
        
        # backpropagate through fully connected layers
        # only evaluates with > 1 FC layer
        for i in range(2, len(self.layers)):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.act_fn.fn_prime(zs[-i])
            del_b[-i] = delta
            del_w[-i] = np.dot(delta, acts[-i-1].transpose())
        
        delta = unravel(np.dot((self.weights[0]).transpose(), delta) * self.act_fn.fn_prime(zs[-self.depth]), self.feat_maps, x1, y1)
        delta1 = [np.zeros((x0,y0)) for i in range(self.feat_maps)]
        
        for i in range(self.feat_maps):
            for j in range(x0):
                for k in range(y0):
                    delta1[i][j][k] = delta[i][int(j/self.dim_pool[0])][int(k/self.dim_pool[1])] * self.act_fn.fn_prime(zs[0][i][j][k]) * \
                     acts[0][i][j][k] / np.sqrt((sum([acts[0][i][int(j/self.dim_pool[0])*self.dim_pool[0]+u][int(k/self.dim_pool[1])*self.dim_pool[1] + v]**2 \
                     for u in range(self.dim_pool[0]) for v in range(self.dim_pool[1])])))
            del_conv_b[i] = sum(u for v in delta1[i] for u in v)
            for j in range(self.dim_filt[0]):
                for k in range(self.dim_filt[1]):
                    del_conv_w[i][j][k] = sum(delta1[i][l][m]*x[l+j][m+k] for l in range(x0) for m in range(y0))
            
        return (del_conv_w, del_w, del_conv_b, del_b)
    
    
    def fit(self, training, eta=.5, batch_size=10, epochs=50, lmbda=0.0, test=None, graph=True):
        # stochastic gradient descent with momentum and regularization in
        # in epochs on minibatches of size batch_size on tuples of training
        #  data (training) with learning rate eta, regularity parameter lmbda
        lc = []
        tc=[]
        n=len(training)
        for i in range(epochs):
            print("Epoch {}".format(i+1))
            random.shuffle(training)
            batches = [training[k:k+batch_size] for k in range(0, n, batch_size)]
            c = 0
            for batch in batches:
                print(c)
                self.batch_update(batch, eta, lmbda, len(training))
                c+=1
            training_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in training]
            lc.append(sum(int(x==y) for (x, y) in training_results)/len(training_results))
            if test:
                test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test]
                tc.append(sum(int(x==y) for (x, y) in test_results)/len(test_results))
        xs = [i for i in range(epochs)]
        if graph:
            plt.plot(xs, lc, color='green')
            if test:
                plt.plot(xs, tc, color='red')
            plt.show()
            
    def batch_update(self, batch, eta, lmbda, n):
        # updates weights and biases in batches
        
        del_b = [np.zeros(bias.shape)  for bias in self.biases]
        del_w = [np.zeros(weight.shape) for weight in self.weights]
        del_cb = [0 for i  in self.conv_bs]
        del_cw = [np.zeros(weight.shape) for weight in self.conv_ws]
        
        for x, y in batch:
            delta_cw, delta_w, delta_cb, delta_b = self.backprop(x, y)
            del_b = [db + dtb for db, dtb in zip(del_b, delta_b)]
            del_w = [dw + dtw for dw, dtw in zip(del_w, delta_w)]
            del_cb = [db + dtb for db, dtb in zip(del_cb, delta_cb)]
            del_cw = [dw + dtw for dw, dtw in zip(del_cw, delta_cw)]
        self.biases = [bias - db*eta/len(batch) for bias, db in zip(self.biases, del_b)] 
        self.weights = [weight*(1-lmbda*eta/n)-dw*eta/len(batch) for weight, dw in zip(self.weights, del_w)]  # final batch not guaranteed to be of size batch_size...
        self.conv_bs = [bias - db*eta/len(batch) for bias, db in zip(self.conv_bs, del_cb)] 
        self.conv_ws = [weight*(1-lmbda*eta/n)-dw*eta/len(batch) for weight, dw in zip(self.conv_ws, del_cw)]  # final batch not guaranteed to be of size batch_size...