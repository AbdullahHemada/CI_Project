
# =========================
# Base Layer Class
# =========================
class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


# =========================
# Dense Layer
# =========================
class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1  # small weights
        self.b = np.zeros((1, output_dim))                     # zero bias

    def forward(self, x):
        self.x = x                                             # save input
        return x @ self.W + self.b                             # Wx + b

    def backward(self, grad):
        # grad = dL/dZ (incoming gradient)

        # Compute gradients
        self.dW = self.x.T @ grad                              # dL/dW
        self.db = np.sum(grad, axis=0, keepdims=True)          # dL/db

        # Return gradient for next layer (dL/dX)
        return grad @ self.W.T
