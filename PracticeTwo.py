"""
Practice 2 is about the following:
-> Gaussian elimination
-> Back substitution
-> Computing Cholesky Decomposition
-> Solving the system L L^T x = b where L is a lower triangular matrix
-> Setting up the linear system describing the tomographic reconstruction
-> Computing tomographic image
"""

import numpy as np
import tomograph

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # creating copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # If the shape of the matrices above are not compatible to each other, then it's going to raise an error!

    r_A, c_A = A.shape  # r_A = rows of A, c_A = columns of A
    c_b = 1  # not used, c_b = columns of b
    r_b = b.shape  # r_b[0] = rows of b

    if r_A != c_A:
        raise ValueError("The matrix is not square!")
    elif r_A != r_b[0]:
        raise ValueError("The number of rows of A must equal the number of rows of b!")
    else:
        pass

    # performing gaussian elimination

    for outer_index in range(c_A - 1):

        if use_pivoting:
            max_of_column = (A[outer_index + 1:, outer_index]).max()
            temp_row = np.where(A[outer_index + 1:, outer_index] == max_of_column)
            A[[outer_index, temp_row[0][0] + (outer_index + 1)]] = A[
                [temp_row[0][0] + (outer_index + 1), outer_index]]  # swapping the needed rows!
            b[[outer_index, temp_row[0][0] + (outer_index + 1)]] = b[
                [temp_row[0][0] + (outer_index + 1), outer_index]]  # swapping the needed rows!

        if np.isclose(np.diag(A)[outer_index], 0):
            if np.allclose(A[(outer_index + 1):, outer_index], 0):
                continue  # In this case, pivoting can't be conducted! All elements below the diagonal entry are zero. We'll go back the top of the loop.
            else:
                if not use_pivoting:
                    raise ValueError("Pivoting is required! The boolean expression of use_pivoting must be True.")
                else:
                    pass

        for inner_index in range(A[(outer_index + 1):, outer_index].size):
            factor = A[outer_index + 1:, outer_index][inner_index] / np.diag(A)[outer_index]
            A[(outer_index + 1) + inner_index] = A[(outer_index + 1) + inner_index] - A[outer_index] * factor
            b[(outer_index + 1) + inner_index] = b[(outer_index + 1) + inner_index] - b[outer_index] * factor

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # If the shape of the matrices above are not compatible to each other, then it's going to raise an error!

    r_A, c_A = A.shape  # r_A = rows of A, c_A = columns of A
    c_b = 1  # not used, c_b = columns of b
    r_b = b.shape  # r_b[0] = rows of b

    if r_A != r_b[0]:
        raise ValueError("The number of rows of A must equal the number of rows of b!")
    else:
        pass

    # initializing solution vector with proper size
    x = np.ones(r_b[0])

    # running the back substitution and filling the solution vector, raising ValueError if no/infinite solutions exist
    for i in range((r_A - 1), -1, -1):

        inside_sigma = 0

        if np.isclose(A[i][i], 0, atol=10 * np.finfo(A.dtype).eps):
            raise ValueError("There are no solutions or infinite solutions!")

        for k in range(i + 1, r_A):
            inside_sigma += A[i][k] * x[k]

        x[i] = (b[i] / A[i][i]) - ((1 / A[i][i]) * inside_sigma)

    return x

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # checking for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if not np.allclose(M, np.transpose(M)):
        raise ValueError("The matrix is not symmetric!")
    else:
        pass

    # building the factorization and raising a ValueError in case of a non-positive definite input matrix
    b = np.zeros(n)

    L = np.zeros((n, n))

    for i in range(0, n):  # i is the outer_index

        for j in range(0, i + 1):  # j is the inner_index

            if i != j:
                sigma_for_non_diagonals = 0

                for k in range(0, j):
                    sigma_for_non_diagonals += L[i][k] * L[j][k]

                if np.isclose(L[j][j], 0):
                    raise ValueError("The matrix is not positive definite!")

                L[i][j] = 1 / L[j][j] * (M[i][j] - sigma_for_non_diagonals)

            else:
                sigma_for_diagonals = 0

                for k in range(0, i):
                    sigma_for_diagonals += np.square(L[i][k])

                if M[i][i] - sigma_for_diagonals < 0:
                    raise ValueError("The matrix is not positive definite!")

                L[i][i] = np.sqrt(M[i][i] - sigma_for_diagonals)

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # checking the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape

    r_b = b.shape  # r_b[0] = rows of b

    if n != m:
        raise ValueError("The matrix is not square!")
    elif n != r_b[0]:
        raise ValueError("The number of rows of L must equal the number of rows of b!")
    elif not np.allclose(L, np.tril(L)):
        raise ValueError("The matrix is not lower triangular!")
    else:
        pass

    # solving the system by forward- and back substitution
    x = np.zeros(m)
    y = np.zeros(m)

    for i in range(0, n):
        inside_sigma = 0

        for k in range(0, i):
            inside_sigma += L[i][k] * y[k]

        y[i] = (1 / L[i][i]) * (b[i] - inside_sigma)

    x = back_substitution(np.transpose(L), y)

    return x

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # initializing the system matrix with proper size
    L = np.zeros((n_rays * n_shots, n_grid * n_grid))
    # initializing the intensity vector
    g = np.zeros(n_rays * n_shots)
    # iterating over equispaced angles, take measurements, and updating the system matrix and sinogram
    # theta = 0  # not used!
    # taking a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    # intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta) #  not used!

    off_set_g = 0
    off_set_L = 0

    for shot in range(len((np.linspace(0, np.pi, n_shots, False)))):

        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays,
                                                                                      np.linspace(0, np.pi, n_shots, False)[
                                                                                          shot])

        for index in range(len(intensities)):
            g[index + off_set_g] = intensities[index]

        for index_two in range(len(ray_indices)):
            L[ray_indices[index_two] + off_set_L][isect_indices[index_two]] = lengths[index_two]

        off_set_g += n_rays
        off_set_L += n_rays

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # solving for tomographic image using the Cholesky solver

    L_A = np.dot(np.transpose(L), L)
    L_B = np.dot(np.transpose(L), g)

    cholesky_L = np.linalg.cholesky(L_A)
    cholesky_vector = solve_cholesky(cholesky_L, L_B)

    # converting the solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    offset_normal = 0

    for i in range(n_grid):

        for j in range(n_grid):

            tim[i][j] = cholesky_vector[j + offset_normal]

        offset_normal += n_grid

    return tim


if __name__ == '__main__':
    print("It's on GitHub!")
