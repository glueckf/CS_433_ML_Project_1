"""Behavioural tests for the six methods in implementations.py.

The official grading tests check a single small example. These check
properties that must hold on any data, and compare against references computed
independently of the implementation (hand-worked values, finite differences,
the pseudo-inverse), so a shared mistake cannot make both sides agree.
"""

import numpy as np
import pytest

import implementations as impl

# Any overflow or invalid-value warning inside a method counts as a failure.
pytestmark = pytest.mark.filterwarnings("error")

N_SAMPLES = 200
N_FEATURES = 4


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def regression_data(rng):
    tx = np.c_[np.ones(N_SAMPLES), rng.normal(size=(N_SAMPLES, N_FEATURES - 1))]
    w_true = rng.normal(size=N_FEATURES)
    y = tx @ w_true + 0.1 * rng.normal(size=N_SAMPLES)
    return y, tx


@pytest.fixture
def classification_data(rng):
    tx = np.c_[np.ones(N_SAMPLES), rng.normal(size=(N_SAMPLES, N_FEATURES - 1))]
    w_true = rng.normal(size=N_FEATURES)
    probabilities = 1 / (1 + np.exp(-(tx @ w_true)))
    y = (rng.random(N_SAMPLES) < probabilities).astype(float)
    return y, tx


def numerical_gradient(f, w, eps=1e-6):
    """Central finite-difference gradient of a scalar function f at w."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = eps
        grad[i] = (f(w + step) - f(w - step)) / (2 * eps)
    return grad


def naive_logistic_loss(y, tx, w):
    """Textbook negative log-likelihood, fine for moderate logits."""
    p = 1 / (1 + np.exp(-(tx @ w)))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- Losses and gradients ---------------------------------------------------


def test_mse_hand_worked_example():
    y = np.array([1.0, 2.0])
    tx = np.eye(2)
    w = np.zeros(2)
    # Errors are [1, 2], so the loss is (1 + 4) / (2 * 2).
    assert impl.compute_mse(y, tx, w) == pytest.approx(1.25)


def test_mse_gradient_matches_finite_differences(regression_data, rng):
    y, tx = regression_data
    w = rng.normal(size=N_FEATURES)
    expected = numerical_gradient(lambda v: impl.compute_mse(y, tx, v), w)
    np.testing.assert_allclose(impl.compute_mse_gradient(y, tx, w), expected)


def test_logistic_loss_matches_textbook_formula(classification_data, rng):
    y, tx = classification_data
    w = rng.normal(size=N_FEATURES)
    expected = naive_logistic_loss(y, tx, w)
    assert impl.compute_logistic_loss(y, tx, w) == pytest.approx(expected)


def test_logistic_loss_is_finite_for_huge_logits(classification_data, rng):
    y, tx = classification_data
    w = rng.normal(size=N_FEATURES)
    assert np.isfinite(impl.compute_logistic_loss(y, 500 * tx, w))


def test_logistic_gradient_matches_finite_differences(classification_data, rng):
    y, tx = classification_data
    w = rng.normal(size=N_FEATURES)
    expected = numerical_gradient(lambda v: impl.compute_logistic_loss(y, tx, v), w)
    np.testing.assert_allclose(
        impl.compute_logistic_gradient(y, tx, w), expected, atol=1e-8
    )


# --- Sigmoid ----------------------------------------------------------------


def test_sigmoid_known_values():
    assert impl.sigmoid(0.0) == 0.5
    assert impl.sigmoid(np.log(3.0)) == pytest.approx(0.75)


def test_sigmoid_is_symmetric():
    t = np.linspace(-20, 20, 41)
    np.testing.assert_allclose(impl.sigmoid(t) + impl.sigmoid(-t), 1.0)


def test_sigmoid_extreme_inputs_stay_finite_and_bounded():
    s = impl.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(s))
    assert s[0] == pytest.approx(0.0)
    assert s[1] == pytest.approx(1.0)


def test_sigmoid_keeps_shape():
    t = np.zeros((3, 2))
    assert impl.sigmoid(t).shape == (3, 2)


# --- Return contract shared by all six methods ------------------------------

METHODS = {
    "mean_squared_error_gd": lambda y, tx, w: impl.mean_squared_error_gd(
        y, tx, w, 10, 0.1
    ),
    "mean_squared_error_sgd": lambda y, tx, w: impl.mean_squared_error_sgd(
        y, tx, w, 10, 0.01
    ),
    "least_squares": lambda y, tx, w: impl.least_squares(y, tx),
    "ridge_regression": lambda y, tx, w: impl.ridge_regression(y, tx, 0.1),
    "logistic_regression": lambda y, tx, w: impl.logistic_regression(y, tx, w, 10, 0.5),
    "reg_logistic_regression": lambda y, tx, w: impl.reg_logistic_regression(
        y, tx, 0.1, w, 10, 0.5
    ),
}

# The loss each method must report for its returned w, without any penalty.
UNPENALISED_LOSS = {
    "mean_squared_error_gd": impl.compute_mse,
    "mean_squared_error_sgd": impl.compute_mse,
    "least_squares": impl.compute_mse,
    "ridge_regression": impl.compute_mse,
    "logistic_regression": naive_logistic_loss,
    "reg_logistic_regression": naive_logistic_loss,
}


@pytest.mark.parametrize("name", METHODS)
def test_returns_1d_weights_and_scalar_loss(name, classification_data):
    y, tx = classification_data
    w, loss = METHODS[name](y, tx, np.zeros(N_FEATURES))
    assert w.shape == (N_FEATURES,)
    assert isinstance(loss, np.floating)
    assert loss.ndim == 0


@pytest.mark.parametrize("name", METHODS)
def test_loss_belongs_to_returned_weights_without_penalty(name, classification_data):
    y, tx = classification_data
    w, loss = METHODS[name](y, tx, np.zeros(N_FEATURES))
    assert loss == pytest.approx(UNPENALISED_LOSS[name](y, tx, w))


@pytest.mark.parametrize("name", METHODS)
def test_initial_weights_are_not_modified(name, classification_data):
    y, tx = classification_data
    initial_w = np.full(N_FEATURES, 0.1)
    METHODS[name](y, tx, initial_w)
    np.testing.assert_array_equal(initial_w, np.full(N_FEATURES, 0.1))


ITERATIVE = {
    "mean_squared_error_gd": lambda y, tx, w: impl.mean_squared_error_gd(
        y, tx, w, 0, 0.1
    ),
    "mean_squared_error_sgd": lambda y, tx, w: impl.mean_squared_error_sgd(
        y, tx, w, 0, 0.1
    ),
    "logistic_regression": lambda y, tx, w: impl.logistic_regression(y, tx, w, 0, 0.1),
    "reg_logistic_regression": lambda y, tx, w: impl.reg_logistic_regression(
        y, tx, 0.5, w, 0, 0.1
    ),
}


@pytest.mark.parametrize("name", ITERATIVE)
def test_zero_iterations_returns_initial_weights(name, classification_data, rng):
    y, tx = classification_data
    initial_w = rng.normal(size=N_FEATURES)
    w, loss = ITERATIVE[name](y, tx, initial_w)
    np.testing.assert_array_equal(w, initial_w)
    assert loss == pytest.approx(UNPENALISED_LOSS[name](y, tx, initial_w))


# --- Linear regression ------------------------------------------------------


def test_least_squares_matches_pseudo_inverse(regression_data):
    y, tx = regression_data
    w, _ = impl.least_squares(y, tx)
    np.testing.assert_allclose(w, np.linalg.pinv(tx) @ y)


def test_ridge_without_regularisation_equals_least_squares(regression_data):
    y, tx = regression_data
    w_ridge, _ = impl.ridge_regression(y, tx, 0.0)
    w_ls, _ = impl.least_squares(y, tx)
    np.testing.assert_allclose(w_ridge, w_ls)


def test_ridge_minimises_mse_plus_penalty(regression_data):
    # Ridge minimises MSE(w) + lambda * ||w||^2, so that objective must have a
    # zero gradient at the returned weights.
    y, tx = regression_data
    lambda_ = 0.3
    w, _ = impl.ridge_regression(y, tx, lambda_)

    def objective(v):
        return impl.compute_mse(y, tx, v) + lambda_ * v @ v

    np.testing.assert_allclose(numerical_gradient(objective, w), 0.0, atol=1e-6)


def test_ridge_shrinks_weights_as_lambda_grows(regression_data):
    y, tx = regression_data
    norms = [np.linalg.norm(impl.ridge_regression(y, tx, lam)[0]) for lam in (0, 1, 10)]
    assert norms[0] > norms[1] > norms[2]


def test_gradient_descent_converges_to_least_squares(regression_data):
    y, tx = regression_data
    w_gd, _ = impl.mean_squared_error_gd(y, tx, np.zeros(N_FEATURES), 2000, 0.5)
    w_ls, _ = impl.least_squares(y, tx)
    np.testing.assert_allclose(w_gd, w_ls, atol=1e-6)


def test_sgd_on_one_sample_equals_gradient_descent(regression_data):
    # With a single datapoint, the stochastic gradient is the full gradient.
    y, tx = regression_data
    initial_w = np.zeros(N_FEATURES)
    w_sgd, _ = impl.mean_squared_error_sgd(y[:1], tx[:1], initial_w, 20, 0.1)
    w_gd, _ = impl.mean_squared_error_gd(y[:1], tx[:1], initial_w, 20, 0.1)
    np.testing.assert_allclose(w_sgd, w_gd)


def test_sgd_is_reproducible_with_a_seed(regression_data):
    y, tx = regression_data
    np.random.seed(1)
    first, _ = impl.mean_squared_error_sgd(y, tx, np.zeros(N_FEATURES), 200, 0.01)
    np.random.seed(1)
    second, _ = impl.mean_squared_error_sgd(y, tx, np.zeros(N_FEATURES), 200, 0.01)
    np.testing.assert_array_equal(first, second)


def test_sgd_approaches_least_squares_loss(regression_data):
    y, tx = regression_data
    np.random.seed(0)
    _, loss_sgd = impl.mean_squared_error_sgd(y, tx, np.zeros(N_FEATURES), 5000, 0.01)
    _, loss_ls = impl.least_squares(y, tx)
    assert loss_sgd < 1.5 * loss_ls


# --- Logistic regression ----------------------------------------------------


def test_logistic_regression_reaches_a_stationary_point(classification_data):
    y, tx = classification_data
    w, _ = impl.logistic_regression(y, tx, np.zeros(N_FEATURES), 3000, 1.0)
    gradient = numerical_gradient(lambda v: naive_logistic_loss(y, tx, v), w)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-5)


def test_reg_logistic_without_regularisation_equals_logistic(classification_data):
    y, tx = classification_data
    initial_w = np.zeros(N_FEATURES)
    w_plain, _ = impl.logistic_regression(y, tx, initial_w, 100, 0.5)
    w_reg, _ = impl.reg_logistic_regression(y, tx, 0.0, initial_w, 100, 0.5)
    np.testing.assert_allclose(w_plain, w_reg)


def test_reg_logistic_step_uses_lambda_times_squared_norm(classification_data, rng):
    # One step must follow the gradient of loss(w) + lambda * ||w||^2.
    y, tx = classification_data
    lambda_, gamma = 0.2, 0.1
    initial_w = rng.normal(size=N_FEATURES)
    w, _ = impl.reg_logistic_regression(y, tx, lambda_, initial_w, 1, gamma)

    def objective(v):
        return naive_logistic_loss(y, tx, v) + lambda_ * v @ v

    expected = initial_w - gamma * numerical_gradient(objective, initial_w)
    np.testing.assert_allclose(w, expected, atol=1e-8)


def test_reg_logistic_shrinks_weights(classification_data):
    y, tx = classification_data
    initial_w = np.zeros(N_FEATURES)
    w_weak, _ = impl.reg_logistic_regression(y, tx, 0.0, initial_w, 500, 0.5)
    w_strong, _ = impl.reg_logistic_regression(y, tx, 0.5, initial_w, 500, 0.5)
    assert np.linalg.norm(w_strong) < np.linalg.norm(w_weak)
