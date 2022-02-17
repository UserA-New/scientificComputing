import numpy as np

"""
Practice 6 is about the following:
-> Finding a root of function f(x) in (lival, rival) with bisection method.
-> Finding a root of f(x)/f(z) starting from start using Newton's method.
-> Generating a Newton fractal for a given function and sampling data.
-> Calculating the area of the given surface represented as triangles in f.
-> Calculating the area gradient of the given surface represented as triangles in f.
-> Calculating the minimal area surface for the given triangles in v/f and boundary representation in c.
"""


def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0,
                        n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # setting a meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    if ival_size == -1.0:
        ival_size = 10 * np.finfo(np.float64).eps

    # initialize iteration
    lival = - 4
    rival = 1500
    fl = f(lival)
    fr = f(rival)

    fl_positive = False
    fr_positive = False

    # making sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))
    n = 0
    # TODO: loop until final interval is found, stop if max iterations are reached
    for count in range(n_iters_max):

        if fr > 0:
            fr_positive = True
        elif fl > 0:
            fl_positive = True

        candidate = (lival + rival) / 2
        fc = f(candidate)

        if abs(fc) < ival_size:
            break

        if fc > 0:
            if fr_positive:
                rival = candidate
            elif fl_positive:
                lival = candidate
        elif fc < 0:
            if fr_positive:
                lival = candidate
            elif fl_positive:
                rival = candidate
        n += 1

    # calculating final approximation to root
    root = candidate
    print(n)
    return root


def func_f(x):
    return x ** 3 - 2 * x + 2  # -1.76929235423863


def deri_f(x):
    return 3 * x ** 2 - 2


def func_g(x):
    return 6 * x / (x ** 2 + 1)


def deri_g(x):
    return 6 * (1 - x ** 2) / (x ** 2 + 1) ** 2


def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert (n_iters_max > 0)

    # initializing root with start value
    root = start

    # choosing a meaningful convergence criterion eps, e.g 10 * eps
    conCri = 10 * np.finfo(np.float64).eps

    n_iterations = 0
    # looping until the convergence criterion eps is met
    while True:

        n_iterations += 1

        # returning root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid
        # divergence)
        if abs(df(root)) < conCri or abs(root) > 1e5:
            return root, (n_iters_max + 1)

        # updating root value and function/dfunction values
        ex_root = root
        root = root - f(root) / df(root)

        # avoiding infinite loops and return (root, n_iters_max+1)
        if n_iterations == n_iters_max:
            return root, (n_iters_max + 1)

        # checking if root ≈ ex_root
        if np.isclose(root, ex_root):
            break

    return root, n_iterations


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray,
                            n_iters_max: int = 20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maximum number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it.
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)
    # iterating over sampling grid
    for r_i in range(sampling.shape[0]):
        for c_i in range(sampling.shape[1]):
            # running Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max
            # iterations)
            root = find_root_newton(f, df, sampling[r_i][c_i], n_iters_max)
            # determining the index of the closest root from the roots array. The functions np.argmin and np.tile
            # could be helpful.
            root_matrix = np.array(root[0])
            root_matrix = np.tile(root_matrix, roots.size)
            index = np.argmin(np.abs(roots - root_matrix))
            # writing the index and the number of needed iterations to the result
            result[r_i, c_i] = np.array([index, root[1]])

    return result


def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """
    # initialize area
    area = 0.0
    # iterating over all triangles and sum up their area
    for v_i in f:
        vector_0 = v[v_i[0]] - v[v_i[1]]
        vector_1 = v[v_i[0]] - v[v_i[2]]
        area += 1 / 2 * np.linalg.norm(np.cross(vector_0, vector_1))
    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)
    # iterating over all triangles and sum up the vertices gradients
    for v_i in f:
        v_0 = v[v_i[0]] - v[v_i[1]]
        v_1 = v[v_i[2]] - v[v_i[0]]
        v_2 = v[v_i[1]] - v[v_i[2]]

        normal = np.cross(v_2, v_1) / np.linalg.norm(np.cross(v_2, v_1))

        gradient[v_i[0]] += 1 / 2 * np.cross(normal, v_2)
        gradient[v_i[1]] += 1 / 2 * np.cross(normal, v_1)
        gradient[v_i[2]] += 1 / 2 * np.cross(normal, v_0)

    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float = 1e-6, ste=1.0, fac=0.5) -> (
        bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """

    # calculating the gradient and area before changing the surface
    gradient = surface_area_gradient(v, f)
    area = surface_area(v, f)

    # calculating the indices of vertices whose position can be changed
    set_indices = np.unique(f.flatten())
    for element in c:
        index = np.argwhere(set_indices == element)
        set_indices = np.delete(set_indices, index)
    the_mutable = set_indices

    # finding a suitable step size so that area can be decreased, don't change v yet
    step = 1  # 0.01 or 0.1 (learning rate)

    v_new = v.copy()
    for index in the_mutable:
        v_new[index] = v[index] + step * gradient[index]
    test_area = surface_area(v_new, f)

    while area - test_area <= epsilon:
        step *= fac
        for index in the_mutable:
            v_new[index] = v[index] + step * gradient[index]
        test_area = surface_area(v_new, f)
        if step <= epsilon:
            break

    # now updating vertex positions in v
    for index in the_mutable:
        v[index] = v[index] + step * gradient[index]

    new_area = surface_area(v, f)
    # checking if new area differs only epsilon from old area
    if np.abs(new_area - area) > epsilon:
        return False, area, v, gradient
    else:
        # Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)
        return True, area, v, gradient


if __name__ == '__main__':
    print("It's on GitHub!")
