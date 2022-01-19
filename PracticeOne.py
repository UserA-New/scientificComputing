"""
Practice 1 is about the following:

-> Matrix multiplication
-> Comparing two matrix multiplication functions, namely between the one I implemented and Numpy's.
-> Machine epsilon
-> Rotation matrix
-> Inverse of the rotation matrix
"""

import numpy as np

from lib import timedcall, plot_2d


def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate product of two matrices a * b.

    Arguments:
    a : first matrix
    b : second matrix

    Return:
    c : matrix product a * b

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -

    Forbidden: numpy.dot, numpy.matrix, numpy.einsum
    """

    n, m_a = a.shape
    m_b, p = b.shape

    # If the shape of the matrices above are not compatible to each other, then it's going to raise an error!
    if m_a != m_b:
        raise ValueError("The number of columns of the 1st matrix must equal the number of rows of the 2nd matrix!")

    # initializing result matrix with zeros
    c = np.zeros((n, p))

    # computing matrix product without the usage of numpy.dot()
    for i in range(n):
        for j in range(p):
            sum_of_the_block = 0
            for k in range(m_a):
                sum_of_the_block += a[i][k] * b[k][j]
            c[i][j] = sum_of_the_block

    return c


def compare_multiplication(nmax: int, n: int) -> dict:
    """
    Compare performance of numpy matrix multiplication (np.dot()) and matrix_multiplication.

    Arguments:
    nmax : maximum matrix size to be tested
    n : step size for matrix sizes

    Return:
    tr_dict : numpy and matrix_multiplication timings and results {"timing_numpy": [numpy_timings],
    "timing_mat_mult": [mat_mult_timings], "results_numpy": [numpy_results], "results_mat_mult": [mat_mult_results]}

    Raised Exceptions:
    -

    Side effects:
    Generates performance plots.
    """

    x, y_mat_mult, y_numpy, r_mat_mult, r_numpy = [], [], [], [], []
    tr_dict = dict(timing_numpy=y_numpy, timing_mat_mult=y_mat_mult, results_numpy=r_numpy, results_mat_mult=r_mat_mult)

    for m in range(2, nmax, n):
        # creating random mxm matrices a and b
        a = np.random.randn(m, m)
        b = np.random.randn(m, m)

        # executing the functions below and measuring the execution time
        time_mat_mult, result_mat_mult = timedcall(matrix_multiplication, a, b)
        time_numpy, result_numpy = timedcall(np.dot, a, b)

        # adding calculated values to lists
        x.append(m)
        y_numpy.append(time_numpy)
        y_mat_mult.append(time_mat_mult)
        r_numpy.append(result_numpy)
        r_mat_mult.append(result_mat_mult)

    # ploting the computed data
    plot_2d(x_data=x, y_data=[y_mat_mult, y_numpy], labels=["matrix_mult", "numpy"],
            title="NumPy vs. for-loop matrix multiplication",
            x_axis="Matrix size", y_axis="Time", x_range=[2, nmax])

    return tr_dict


def machine_epsilon(fp_format: np.dtype) -> np.number:
    """
    Calculate the machine precision for the given floating point type.

    Arguments:
    fp_format: floating point format, e.g. float32 or float64

    Return:
    eps : calculated machine precision

    Raised Exceptions:
    -

    Side Effects:
    Prints out iteration values.

    Forbidden:
numpy.finfo
    """

    # creating the epsilon element with correct initial value and data format fp_format
    eps = fp_format.type(1.0)
    not_eps = fp_format.type(1.0)

    # creating necessary variables for iteration
    one = fp_format.type(1.0)
    two = fp_format.type(2.0)
    i = 1

    print('  i  |       2^(-i)        |  1 + 2^(-i)  ')
    print('  ----------------------------------------')

    # determining the machine precision without the use of numpy.finfo()

    while fp_format.type(not_eps) + fp_format.type(1) != fp_format.type(1):
        eps = fp_format.type(not_eps)
        not_eps = fp_format.type(not_eps) / fp_format.type(2)

    i = np.log2(eps)

    print('{0:4.0f} |  {1:16.8e}   | equal 1'.format(i, eps))
    return eps


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2x2 rotation matrix around angle theta.

    Arguments:
    theta : rotation angle (in degrees)

    Return:
    r : rotation matrix

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # creating an empty matrix
    r = np.zeros((2, 2))

    # converting the angle to radians

    theta_in_radians = (theta * np.pi)/180

    # calculating the diagonal terms of matrix

    for i in range(2):
        r[i][i] = np.cos(theta_in_radians)

    # off-diagonal terms of matrix

    r[0][1] = -1 * np.sin(theta_in_radians)

    r[1][0] = np.sin(theta_in_radians)

    return r


def inverse_rotation(theta: float) -> np.ndarray:
    """
    Compute inverse of the 2d rotation matrix that rotates a 
    given vector by theta.
    
    Arguments:
    theta: rotation angle
    
    Return:
    Inverse of the rotation matrix

    Forbidden: numpy.linalg.inv, numpy.linalg.solve
    """

    # computing inverse rotation matrix

    # A note: Rotation matrices are orthogonal, hence A^(-1) = A^T

    m = np.zeros((2, 2))

    theta_in_radians = (theta * np.pi)/180

    for i in range(2):
        m[i][i] = np.cos(theta_in_radians)

    m[0][1] = -1 * np.sin(theta_in_radians)

    m[1][0] = np.sin(theta_in_radians)

    # transposing...

    stored_block = m[0][1]
    m[0][1] = m[1][0]
    m[1][0] = stored_block

    return m


if __name__ == '__main__':
    print("It's on GitHub!")
