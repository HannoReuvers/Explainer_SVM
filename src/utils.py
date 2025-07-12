import numpy as np
from qpsolvers import solve_qp

def logistic_function(x: np.array):
    """
    Apply the standard logistic function element-wise.

    :param x: Input array
    :return out: The output array, the standard logistic function of each element in x.
    """

    return 1.0/(1+np.exp(-x))


def generate_logistic_data(b, theta_1, theta_2, sample_size=20):
    """
    Generate example data from a logistic regression. The regressors are uniformly distributed on [0,1]x[-1,1]. The classes take
    values in the set {-1, 1} depending on the input parameters.

    :param b: Bias parameter
    :param theta_1: Coefficient multiplying the regressor x_1
    :param theta_2: Coefficient multiplying the regressor x_2
    :return x_1: Generated regressor x_1
    :return x_2: Generated regressor x_2
    :return y: Array containing labels
    """

    # Generate coordinates of points
    rng = np.random.default_rng(1234)
    x_1 = rng.uniform( 0, 1, size=(sample_size, 1))
    x_2 = rng.uniform(-1, 1, size=(sample_size, 1))

    # Determine class labels
    y = -1+2.0*(rng.uniform( 0, 1, size=(sample_size, 1)) <= logistic_function(b+theta_1*x_1+theta_2*x_2) )

    return x_1, x_2, y.astype(int)


def determine_separating_hyperplane(x_1, x_2, y):
    """
    Solve the quadratic program (QP) that provides the separating hyperplane. The implementation assumes that the feature space is two-dimensional
    and the separating hyperplane is thus a line.

    :param x_1: The vector containing the values for the first feature
    :param x_2: The vector containing the values for the second feature
    :param y: Array containing labels
    :return slope: The slope of the separating hyperplane (= line in 2D)
    :return intercept: The intercept of the separating hyperplane (= line in 2D)
    """

    # Number of features
    K = 2

    # Create auxiliary matrix
    X = np.column_stack((x_1,x_2))
    ones = np.full((len(y), 1), 1)   
    X_total = np.column_stack((X,ones))

    # Auxiliary matrix
    X_total = np.column_stack((X,ones))

    # QP-specific quantities
    P = np.block([
        [np.identity(K),    np.zeros((K, 1))],
        [np.ones((1, K)),   np.zeros((1,1))]])
    q = np.zeros((3, 1))
    G = -y*X_total
    h = -ones

    # Solve QP
    x = solve_qp(P, q, G, h, solver="clarabel")

    # Translate QP output to slope and intercept of the separating hyperplane (= line in 2D)
    slope = -x[0]/x[1]
    intercept = -x[2]/x[1]

    return slope, intercept