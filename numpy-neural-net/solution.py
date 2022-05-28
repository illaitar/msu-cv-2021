from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """
        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            inertia = np.zeros(parameter_shape)
            inertia = self.momentum * inertia + self.lr * parameter_grad
            parameter2 = parameter - inertia
            return parameter2
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        res = inputs.copy()
        res[res < 0] = 0
        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        res = np.array(grad_outputs, copy=True)
        res[self.forward_inputs < 0] = 0
        return res
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        ress = []
        for batch in inputs:
            res = np.exp(batch - np.max(batch))
            ress.append( res / res.sum())
        return np.array(ress)
        # your code here /\
        
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        y = self.forward_outputs
        n, d = grad_outputs.shape[:2]
        new_grad =np.empty(y.shape)
        for i in range(n):
            y_t = y[i].reshape(-1,1)
            new_grad[i] = grad_outputs[i] * y[i] - np.matmul(y_t, np.matmul(np.transpose(y_t),  grad_outputs[i]))
        return new_grad
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        res = []
        for i in range(inputs.shape[0]):
            res.append(inputs[i] @ self.weights + self.biases)
        return np.array(res)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/

        self.biases_grad = np.ravel(np.sum(grad_outputs, axis=0))/ len(grad_outputs)
        self.weights_grad = 0
        self.weights_grad = self.forward_inputs.T @ grad_outputs / len(grad_outputs)
        res = []
        for i in range(grad_outputs.shape[0]):
            res.append(grad_outputs[i] @ self.weights.T)
        return np.array(res)
        
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n,)), loss scalars for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        res = -np.sum(y_gt*np.log(y_pred), axis = 1)
        return res
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), gradient loss to y_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        y_pr = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                y_pr[i][j] = max(y_pred[i][j], eps)
        return -np.divide(y_gt,y_pr)
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.04))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape= (784,), units = 20))
    model.add(ReLU())
    model.add(Dense(units = 20))
    model.add(ReLU())
    model.add(Dense(units = 10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 5, 2)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    ker = kernels.copy()
    n, d, ih, iw = inputs.shape
    c, d, kh, kw = kernels.shape
    owidth = iw + 2 * padding - kw + 1
    oheight = ih + 2 * padding - kh + 1
    inputs = np.pad(inputs, ((0,0),(0,0),(padding,padding),(padding,padding)))
    ker = np.flip(ker,axis = (2,3))
    res = np.zeros((n,c,oheight,owidth))
    for t in range(oheight):
        for s in range(owidth):
            res[...,t,s] = (inputs[...,t:t + kh,s:s + kw].transpose(2,3,0,1)[...,np.newaxis,:] @ ker.transpose(2,3,1,0)[:,:,np.newaxis,:,:]).squeeze(3).transpose(2,3,0,1).reshape(n,c, -1).sum(axis = -1)
    return res
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        res = convolve(inputs,self.kernels,(self.kernels.shape[2]-1)//2)
        for i in range(res.shape[1]):
            res[:, i, :, :] += self.biases[i]
        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        p = self.kernel_size // 2
        p_inv = self.kernel_size - p - 1
        X = self.forward_inputs
        X_hatT = np.flip(X, axis=(2,3)).transpose(1,0,2,3)
        self.biases_grad = np.sum(grad_outputs, axis = (2,3)).mean(axis=0)
        self.kernels_grad = convolve(X_hatT, grad_outputs.transpose(1,0,2,3), p).transpose(1,0,2,3) / grad_outputs.shape[0]
        res = convolve(grad_outputs, np.flip(self.kernels,axis=(2,3)).transpose(1,0,2,3),p_inv)
        return res
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"
        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        n, d, ih, iw = inputs.shape
        p = self.pool_size
        oh, ow = ih // p, iw // p
        outputs = np.zeros((n, d, oh, ow))
        self.cache = np.zeros((n, d, ih, iw))
        for i in range(oh):
            for j in range(ow):
                if self.pool_mode == 'max':
                    outputs[:, :, i, j] = np.max(inputs[:, :, i*p:(i+1)*p, j*p:(j+1)*p], axis=(2,3))
                    arguments = np.argmax(inputs[:, :, i*p:(i+1)*p, j*p:(j+1)*p].reshape(n, d, -1), axis=2)
                    for k in range(n):
                        for l in range(d):
                            self.cache[k, l, i*p + arguments[k, l] // p, j*p + arguments[k, l] % p] = 1
                else:
                    outputs[:, :, i, j] = np.mean(inputs[:, :, i*p:(i+1)*p, j*p:(j+1)*p], axis=(2,3))
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        n, d, oh, ow = grad_outputs.shape
        p = self.pool_size
        ih, iw = oh * p, ow * p
        dLdI = np.zeros((n, d, ih, iw))
        for i in range(oh):
            for j in range(ow):
                if self.pool_mode == 'max':
                    dLdI[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = grad_outputs[:, :, i, j][..., None, None]
                else:
                    dLdI[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = grad_outputs[:, :, i, j][..., None, None] / (self.pool_size**2)
        if self.pool_mode == 'max':
            dLdI = dLdI * self.cache
        return dLdI
        # your code here /\

# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None
        self.cache_std = None
        self.cache_center = None
        self.cache_norm = None
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        x = inputs.copy()
        if self.is_training:
            mean = np.mean(x,axis=(0,2,3),keepdims=True)
            var = np.var(x,axis=(0,2,3),keepdims=True)
            self.cache_center = x - mean
            self.cache_std = 1 / np.sqrt(var + eps)
            self.cache_norm = self.cache_center * self.cache_std
            x =  self.cache_norm.copy()
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            for i in range(self.gamma.shape[0]):
                x[:,i,:,:] = (x[:,i,:,:] - self.running_mean[i]) / np.sqrt(self.running_var[i] + eps)
        for i in range(self.gamma.shape[0]):
            x[:,i,:,:] = x[:,i,:,:]*self.gamma[i] + self.beta[i]

        return x
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        self.gamma_grad = np.sum(grad_outputs * self.cache_norm, axis=(0, 2, 3))
        self.gamma_grad = self.gamma_grad.reshape(self.gamma.shape) / grad_outputs.shape[0]
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3)).reshape(self.beta.shape) / grad_outputs.shape[0]
        dl_norm = grad_outputs.copy() * self.gamma[np.newaxis, ..., np.newaxis, np.newaxis]
        mean_ext = np.mean(dl_norm * self.cache_norm, axis=(0, 2, 3), keepdims=True)
        mean_rt = np.mean(dl_norm, axis=(0, 2, 3), keepdims=True)
        input_dl = (dl_norm - mean_ext * self.cache_norm - mean_rt) * self.cache_std
        return input_dl
        # your code here /\
# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
 
        self.output_shape = (np.prod(self.input_shape),)
        self.shape_in = None
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values
 
            :return: np.array((n, (d * h * w))), output values
 
                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        self.shape_in = inputs.shape[1:4]
        res = []
        for i in range(inputs.shape[0]):
            res.append(np.ravel(inputs[i]))
        return np.array(res)
        # your code here /\
 
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs
 
            :return: np.array((n, d, h, w)), dLoss/dInputs
 
                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        res = []
        #print(self.output_shape)
        for i in range(grad_outputs.shape[0]):
            res.append(grad_outputs[i].reshape(self.shape_in))
        return np.array(res)
        # your code here /\
 
 
# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None
 
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values
 
            :return: np.array((n, d)), output values
 
                n - batch size
                d - number of units
        """
        # your code here \/
        self.forward_mask = np.random.uniform(0,1,size = inputs.shape)
 
        if self.is_training:
            M = self.forward_mask.copy()
            M[M>self.p] = 1
            M[M<self.p] = 0
            return inputs*M
        else:
            M = self.forward_mask.copy()
            return inputs*M
        # your code here /\
 
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs
 
            :return: np.array((n, d)), dLoss/dInputs
 
                n - batch size
                d - number of units
        """
        # your code here \/
        M = self.forward_mask.copy()
        M[M>self.p] = 1
        M[M<self.p] = 0
        return grad_outputs*M
        # your code here /\
 
 
# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
#     import os
#     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.09))
 
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(output_channels=32, kernel_size=3, input_shape=(3, 32, 32)))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Conv2D(output_channels=32, kernel_size=3))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_mode='max'))
    model.add(Pooling2D(pool_mode='avg'))
    model.add(Flatten())
    model.add(Dense(units = 64))
    model.add(ReLU())
    model.add(Dense(units = 10))
    model.add(Softmax())
    print(model)
 
    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 32, 3)
 
    # your code here /\
    return model
 
# ============================================================================
