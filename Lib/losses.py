# =========================
# Loss: Mean Squared Error
# =========================
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return (2 / y_true.shape[0]) * (y_pred - y_true)

