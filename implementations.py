"""Implementations of the six required machine learning methods."""

import numpy as np


def compute_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.floating:
    """Compute the mean squared error, with the factor 1/2.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).
        w: Weights, shape (D,).

    Returns:
        The MSE as a NumPy scalar.
    """
    pred = tx.dot(w)
    err = y - pred
    return err.dot(err) / (2 * len(err))


def compute_mse_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the gradient of the MSE with respect to w.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).
        w: Weights, shape (D,).

    Returns:
        The gradient, shape (D,).
    """
    pred = tx.dot(w)
    err = y - pred
    return -tx.T.dot(err) / len(y)


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function elementwise, without overflow.

    Args:
        t: Input array of any shape.

    Returns:
        sigmoid(t), same shape as t.
    """
    t = np.asarray(t, dtype=float)
    z = np.exp(-np.abs(t))
    return np.where(t >= 0, 1 / (1 + z), z / (1 + z))


def compute_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.floating:
    """Compute the mean negative log-likelihood of logistic regression.

    Args:
        y: Labels in {0, 1}, shape (N,).
        tx: Features, shape (N, D).
        w: Weights, shape (D,).

    Returns:
        The logistic loss as a NumPy scalar.
    """
    pred = tx.dot(w)
    softplus = np.maximum(0, pred) + np.log1p(np.exp(-np.abs(pred)))
    return np.mean(-y * pred + softplus)


def compute_logistic_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the logistic loss with respect to w.

    Args:
        y: Labels in {0, 1}, shape (N,).
        tx: Features, shape (N, D).
        w: Weights, shape (D,).

    Returns:
        The gradient, shape (D,).
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, np.floating]:
    """Linear regression using gradient descent.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of iterations.
        gamma: Step size.

    Returns:
        (w, loss): the final weights, shape (D,), and their MSE.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, np.floating]:
    """Linear regression using stochastic gradient descent, batch size 1.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of iterations.
        gamma: Step size.

    Returns:
        (w, loss): the final weights, shape (D,), and their MSE on all data.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        gradient = compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, np.floating]:
    """Least squares regression using the normal equations.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).

    Returns:
        (w, loss): the optimal weights, shape (D,), and their MSE.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> tuple[np.ndarray, np.floating]:
    """Ridge regression using the normal equations.

    Args:
        y: Targets, shape (N,).
        tx: Features, shape (N, D).
        lambda_: Regularization strength.

    Returns:
        (w, loss): the optimal weights, shape (D,), and their MSE without
        the penalty term.
    """
    a = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, np.floating]:
    """Logistic regression using gradient descent.

    Args:
        y: Labels in {0, 1}, shape (N,).
        tx: Features, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of iterations.
        gamma: Step size.

    Returns:
        (w, loss): the final weights, shape (D,), and their logistic loss.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, np.floating]:
    """Regularized logistic regression using gradient descent.

    Args:
        y: Labels in {0, 1}, shape (N,).
        tx: Features, shape (N, D).
        lambda_: Regularization strength.
        initial_w: Initial weights, shape (D,).
        max_iters: Number of iterations.
        gamma: Step size.

    Returns:
        (w, loss): the final weights, shape (D,), and their logistic loss
        without the penalty term.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        penalty_grad = 2 * lambda_ * w
        gradient = compute_logistic_gradient(y, tx, w) + penalty_grad
        w = w - gamma * gradient

    loss = compute_logistic_loss(y, tx, w)
    return w, loss
