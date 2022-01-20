"""
Practice 4 is about the following:
-> Lagrange interpolation
-> Hermite cubic interpolation
-> Natural cubic interpolation
-> Periodic cubic interpolation
"""

import numpy as np


def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # generating the Lagrange base polynomials and interpolation polynomial

    number_of_points = x.size
    vx = np.poly1d([1, 0])  # vx : variable x

    for i in range(number_of_points):

        product = 1

        for j in range(number_of_points):
            if i != j:
                product = product * (vx - x[j]) / (x[i] - x[j])

        base_functions.append(product)

    for i in range(number_of_points):
        polynomial += y[i] * base_functions[i]

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)
    number_of_points = x.size

    spline = []
    # computing the piecewise interpolating cubic polynomials
    for i in range(number_of_points - 1):
        A = np.array([[np.power(x[i], 3), np.square(x[i]), x[i], 1],
                      [np.power(x[i + 1], 3), np.square(x[i + 1]), x[i + 1], 1],
                      [3 * np.square(x[i]), 2 * x[i], 1, 0],
                      [3 * np.square(x[i + 1]), 2 * x[i + 1], 1, 0]])

        b = np.array([y[i], y[i + 1], yp[i], yp[i + 1]])

        c = np.matmul(np.linalg.inv(A), b)

        fx = np.poly1d([c[0], c[1], c[2], c[3]])

        spline.append(fx)

    return spline

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    number_of_points = x.size

    # constructing the linear system with natural boundary conditions
    A = np.zeros((4 * number_of_points - 4, 4 * number_of_points - 4))

    for i, element in enumerate(x[0:-1]):
        matrixGmbH = np.array([[np.power(element, 3), np.square(element), element, 1],
                               [np.power(x[i + 1], 3), np.square(x[i + 1]), x[i + 1], 1]])

        matrixCo = np.array([[3 * np.square(x[i + 1]), 2 * x[i + 1], 1, 0, -3 * np.square(x[i + 1]), -2 * x[i + 1], -1, 0],
                             [6 * x[i + 1], 2, 0, 0, -6 * x[i + 1], -2, 0, 0]])

        if i == number_of_points - 2:
            A[(4 * i):(4 * i) + 2, (4 * i):4 * (i + 1)] = matrixGmbH
            break

        A[(4 * i):(4 * i) + 2, (4 * i):4 * (i + 1)] = matrixGmbH
        A[(4 * i) + 2: 4 * (i+1), (4 * i): 4 * (i + 2)] = matrixCo

    #  natural boundary conditions:

    A[-2, 0:2] = np.array([6 * x[0], 2])
    A[-1, -4:-2] = np.array([6 * x[-1], 2])

    b = np.zeros(4 * number_of_points - 4)

    for i, element in enumerate(y[0:-1]):
        b[4 * i] = element
        b[4 * i + 1] = y[i + 1]

    # solving the linear system for the coefficients of the spline
    c = np.linalg.solve(A, b)

    spline = []
    # extracting the local interpolation coefficients from solution
    for i in range(0, number_of_points - 1):
        spline.append(np.poly1d(c[4 * i: 4 * (i + 1)]))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    number_of_points = x.size

    # constructing the linear system with periodic boundary conditions
    A = np.zeros((4 * number_of_points - 4, 4 * number_of_points - 4))

    for i, element in enumerate(x[0:-1]):
        matrixGmbH = np.array([[np.power(element, 3), np.square(element), element, 1],
                               [np.power(x[i + 1], 3), np.square(x[i + 1]), x[i + 1], 1]])

        matrixCo = np.array([[3 * np.square(x[i + 1]), 2 * x[i + 1], 1, 0, -3 * np.square(x[i + 1]), -2 * x[i + 1], -1, 0],
                             [6 * x[i + 1], 2, 0, 0, -6 * x[i + 1], -2, 0, 0]])

        if i == number_of_points - 2:
            A[(4 * i):(4 * i) + 2, (4 * i):4 * (i + 1)] = matrixGmbH
            break

        A[(4 * i):(4 * i) + 2, (4 * i):4 * (i + 1)] = matrixGmbH
        A[(4 * i) + 2: 4 * (i+1), (4 * i): 4 * (i + 2)] = matrixCo

    #  periodic boundary conditions:

    A[-2, 0:4] = np.array([3 * np.square(x[0]), 2 * x[0], 1, 0])
    A[-2, -4:] = np.array([- 3 * np.square(x[-1]), - 2 * x[-1], -1, 0])
    A[-1, 0:2] = np.array([6 * x[0], 2])
    A[-1, -4:-2] = np.array([-6 * x[-1], -2])

    b = np.zeros(4 * number_of_points - 4)

    for i, element in enumerate(y[0:-1]):
        b[4 * i] = element
        b[4 * i + 1] = y[i + 1]

    # solving the linear system for the coefficients of the spline
    c = np.linalg.solve(A, b)

    spline = []
    # extracting the local interpolation coefficients from solution
    for i in range(0, number_of_points - 1):
        spline.append(np.poly1d(c[4 * i: 4 * (i + 1)]))

    return spline


if __name__ == '__main__':
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation(x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("It's on GitHub!")
