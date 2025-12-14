# =========================
# Activation: Tanh
# =========================
class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)   # tanh derivative


# =========================
# Activation: Sigmoid
# =========================
class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * (self.out * (1 - self.out))   # sigmoid derivative
