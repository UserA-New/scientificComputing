"""
Practice 3 is about the following:
-> Power iteration
-> Loading images
-> Setting up data matrix
-> Performing PCA
-> Accumulated energy
   Computing index k so that threshold percent of magnitude of singular values is contained in
   first k singular vectors.
-> Projecting given image set into basis.
-> Performing face recognition
"""

import numpy as np
import lib
import matplotlib as mpl

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix is not a square matrix!")

    # setting epsilon to default value if not set by user
    if epsilon == -1.0:
        epsilon = 10 * np.finfo(float).eps

    # normalized random vector of proper size to initialize iteration
    vector = np.random.randint(7, size=M.shape[0])
    norm = np.linalg.norm(vector)
    vector = vector / norm

    # initializing the residual list and the residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # performing power iteration
    while residual > epsilon:
        residuals.append(residual)  # residual appended
        Mv = np.dot(M, vector)
        eigenvalue = np.linalg.norm(Mv)
        new_vector = Mv / eigenvalue
        residual = np.linalg.norm(new_vector - vector)
        vector = new_vector

    return vector, residuals

def load_images(path: str, file_ending: str = ".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # reading each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()
    files = lib.list_directory(path)
    files.sort()
    for image in files:
        if image.endswith(file_ending):
            image = mpl.image.imread((path + '{}').format(image))
            image = np.asarray(image, dtype=np.float64)
            images.append(image)
        else:
            continue

    # setting the dimensions according to first image in images
    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # initializing data matrix with proper size and data type
    D = np.zeros((len(images), images[0].shape[0] * images[0].shape[1]))

    # adding the flattened images to data matrix
    index = 0
    for vector in images:
        D[index, :] = vector.flatten()
        index += 1

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # subtracting the mean from data / center data at origin
    data_average = np.mean(D, axis=0)
    mean_data = D - np.tile(data_average, (D.shape[0], 1))

    # computing the left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)
    U, S, VT = np.linalg.svd(mean_data, full_matrices=False)

    pcs = VT
    svals = S

    return pcs, svals, data_average


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    k = 0
    # determining k that first k singular values make up threshold percent of magnitude
    singular_values_so_far = np.zeros(singular_values.size)

    while np.sum(singular_values_so_far) < np.sum(singular_values) * threshold:
        singular_values_so_far[k] = singular_values[k]
        k += 1

    lib.plot_singular_values_and_energy(singular_values, k)

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA (data_average for me!)

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # initializing the coefficients array with proper size
    coefficients = np.zeros((len(images), pcs.shape[0]))

    # iterating over images and project each normalized image into principal component basis
    D = setup_data_matrix(images)
    X = D - np.tile(mean_data, (D.shape[0], 1))
    # U, S, VT = np.linalg.svd(X, full_matrices=False)

    for i in range(len(images)):
        coefficients[i] = np.dot(pcs, X[i])

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
        np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    # loading the  test data
    imgs_test = load_images(path_test, ".png")[0]

    # projecting the test data set into eigenbasis
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    # initializing the scores matrix with proper size
    scores = np.zeros((coeffs_train.shape[0], coeffs_test.shape[0]))

    # initializing over all images and calculate pairwise correlation

    for i in range(coeffs_train.shape[0]):
        coeffs_train[i] = coeffs_train[i] / np.linalg.norm(coeffs_train[i])

    for i in range(coeffs_test.shape[0]):
        coeffs_test[i] = coeffs_test[i] / np.linalg.norm(coeffs_test[i])

    not_scores = np.matmul(coeffs_train, np.transpose(coeffs_test))

    for i in range(coeffs_train.shape[0]):
        for j in range(coeffs_test.shape[0]):
            scores[i, j] = np.arccos(not_scores[i, j])

    return scores, imgs_test, coeffs_test


if __name__ == '__main__':
    A = np.random.randn(7, 7)
    A = A.transpose().dot(A)
    L, U = np.linalg.eig(A)
    L[1] = L[0] - 10 ** -3
    A = U.dot(np.diag(L)).dot(U.transpose())
    print()
    np.set_printoptions(precision=16)
    print(A.flatten())

    A = np.array([18.2112344794043359, 0.7559886314903312, 7.2437569750169502,
                  -13.8991061752623271, 4.8768689715057691, -1.318055436971276,
                  -6.7829844205260148, 0.7559886314903312, 7.9204801042364448,
                  1.5378938590357767, 7.1775560914639325, 2.8536549530686015,
                  1.9998683983340397, -5.9532930598376685, 7.2437569750169502,
                  1.5378938590357767, 9.841906218619128, 0.5841092845624152,
                  6.7510103134860797, 4.6111951240722888, -8.9825300821798191,
                  -13.8991061752623271, 7.1775560914639334, 0.5841092845624152,
                  24.2028041177043818, 0.8180957104689988, 6.6087248591945729,
                  -4.1573996873552073, 4.8768689715057691, 2.8536549530686015,
                  6.7510103134860806, 0.8180957104689979, 7.0366782892027206,
                  5.4944303652858073, -9.0773671527609796, -1.318055436971276,
                  1.9998683983340397, 4.6111951240722888, 6.608724859194572,
                  5.4944303652858073, 8.1889694453300805, -7.1176432086570651,
                  -6.7829844205260148, -5.9532930598376685, -8.9825300821798191,
                  -4.1573996873552046, -9.0773671527609796, -7.1176432086570633,
                  13.664209790087753])
    A = A.reshape((7, 7))

    ev, res = power_iteration(A)

    print('ev = ' + str(ev))

    print("It's on GitHub!")
