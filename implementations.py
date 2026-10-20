"""The six machine learning methods required by the CS-433 Project 1 spec.

This file is graded directly: the course test harness clones the repository,
loads this module on its own with a ``SourceFileLoader``, and calls the six
public functions below. That imposes three rules on this file specifically.

1. It must stay self-contained. Do not import from ``mlproject`` here, and do
   not import anything beyond the standard library and NumPy.
2. Every public function must keep its exact name, parameter names, and
   parameter order as given in the project description.
3. Every public function returns ``(w, loss)`` where ``w`` is the final weight
   vector as a 1-D array of shape ``(D,)`` and ``loss`` is the loss of that
   final ``w`` only -- not a history, and not the loss of the best iterate.

The grader asserts ``loss.ndim == 0``, so the loss must be a NumPy scalar
(``np.float64``), never a plain Python ``float`` and never a 1-element array.

Everything that is not one of these six methods -- preprocessing, feature
engineering, cross-validation, metrics -- belongs in the ``mlproject`` package.
"""

import numpy as np


def compute_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.floating:
    """Compute the mean squared error loss.

    The project description fixes the convention ``L = (1 / (2N)) * sum(e**2)``
    with the factor one half, to match the lecture notes.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        w: Weight vector, shape ``(D,)``.

    Returns:
        The mean squared error as a NumPy scalar.
    """
    # Compute the prediction
    pred = tx.dot(w)

    # Compute the error
    err = y - pred

    # Compute MSE
    return err.dot(err) / (2*len(err))


def compute_mse_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the gradient of the mean squared error loss.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        w: Weight vector, shape ``(D,)``.

    Returns:
        The gradient with respect to ``w``, shape ``(D,)``.
    """
    # Compute error 
    pred = tx.dot(w)
    err = y - pred

    return -tx.T.dot(err) / len(y)
    


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid elementwise.

    Use a numerically stable formulation: a naive ``1 / (1 + exp(-t))``
    overflows for large negative ``t`` and will produce warnings and NaNs on
    this dataset.

    Args:
        t: Input array of any shape.

    Returns:
        ``sigmoid(t)``, same shape as ``t``.
    """
    t = np.asarray(t, dtype=float)

    # In (0,1], never overflows
    z = np.exp(-np.abs(t))
    # If t >= 0: z = exp(-t); If t < 0: z = exp(t), Both forms equal sigmoid(t). 
    return np.where(t >= 0, 1/(1+z), z/(1+z))


def compute_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.floating:
    """Compute the average negative log-likelihood for logistic regression.

    Labels must be in ``{0, 1}`` here, not ``{-1, 1}``. Converting the raw
    labels is the caller's job; see ``mlproject.data``.

    Args:
        y: Target values in ``{0, 1}``, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        w: Weight vector, shape ``(D,)``.

    Returns:
        The mean logistic loss as a NumPy scalar.
    """
    pred = tx.dot(w)
    # Stable log(1 + exp(pred)): exp(-|pred|) lies in (0, 1], so it never overflows.
    softplus = np.maximum(0, pred) + np.log1p(np.exp(-np.abs(pred)))
    return np.mean(-y * pred + softplus)

def compute_logistic_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Compute the gradient of the logistic loss.

    Args:
        y: Target values in ``{0, 1}``, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        w: Weight vector, shape ``(D,)``.

    Returns:
        The gradient with respect to ``w``, shape ``(D,)``.
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, np.floating]:
    """Linear regression using full-batch gradient descent.

    Note that ``max_iters=0`` is a valid input and must return ``initial_w``
    together with its loss; the public grading tests check exactly that case.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        initial_w: Starting weight vector, shape ``(D,)``.
        max_iters: Number of gradient descent steps to run.
        gamma: Step size.

    Returns:
        A tuple ``(w, loss)`` with the final weights, shape ``(D,)``, and the
        mean squared error of those final weights as a NumPy scalar.
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
    """Linear regression using stochastic gradient descent.

    The project description mandates a mini-batch size of exactly one: each
    step samples a single datapoint. The returned loss is still the full-batch
    mean squared error of the final weights.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        initial_w: Starting weight vector, shape ``(D,)``.
        max_iters: Number of stochastic gradient steps to run.
        gamma: Step size.

    Returns:
        A tuple ``(w, loss)`` with the final weights, shape ``(D,)``, and the
        mean squared error of those final weights as a NumPy scalar.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        # Slice: n:n+1 (not n) so tx stays 2-D, shape (1, D)
        gradient = compute_mse_gradient(y[n:n+1], tx[n:n+1], w)
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, np.floating]:
    """Least squares regression using the normal equations.

    Solve ``(tx.T @ tx) w = tx.T @ y``. ``numpy.linalg.solve`` is allowed;
    ``numpy.linalg.lstsq`` is explicitly forbidden by the project description.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.

    Returns:
        A tuple ``(w, loss)`` with the optimal weights, shape ``(D,)``, and
        their mean squared error as a NumPy scalar.
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

    Solve ``(tx.T @ tx + 2 * N * lambda_ * I) w = tx.T @ y``. The returned
    loss is the plain mean squared error and must NOT include the penalty
    term.

    Args:
        y: Target values, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        lambda_: Regularisation strength. Trailing underscore because
            ``lambda`` is a reserved word in Python.

    Returns:
        A tuple ``(w, loss)`` with the optimal weights, shape ``(D,)``, and
        their unpenalised mean squared error as a NumPy scalar.
    """
    a = tx.T.dot(tx) + 2*len(y)*lambda_*np.eye(tx.shape[1])
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
    """Logistic regression using full-batch gradient descent.

    Labels must be in ``{0, 1}``.

    Args:
        y: Target values in ``{0, 1}``, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        initial_w: Starting weight vector, shape ``(D,)``.
        max_iters: Number of gradient descent steps to run.
        gamma: Step size.

    Returns:
        A tuple ``(w, loss)`` with the final weights, shape ``(D,)``, and the
        logistic loss of those final weights as a NumPy scalar.
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
    """Regularised logistic regression using full-batch gradient descent.

    The objective adds a penalty ``lambda_ * ||w||**2``, so the gradient picks
    up a ``2 * lambda_ * w`` term. As with ridge regression, the returned loss
    is the plain logistic loss and must NOT include the penalty term.

    Note the parameter order: ``lambda_`` comes before ``initial_w`` here,
    unlike in the unregularised version.

    Args:
        y: Target values in ``{0, 1}``, shape ``(N,)``.
        tx: Feature matrix, shape ``(N, D)``.
        lambda_: Regularisation strength.
        initial_w: Starting weight vector, shape ``(D,)``.
        max_iters: Number of gradient descent steps to run.
        gamma: Step size.

    Returns:
        A tuple ``(w, loss)`` with the final weights, shape ``(D,)``, and the
        unpenalised logistic loss of those final weights as a NumPy scalar.
    """