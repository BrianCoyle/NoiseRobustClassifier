from nisqai.data._cdata import CData, LabeledCData
import numpy as np
from nisqai.visual import scatter

def grid_pts(nx=12, ny=12):
    """
    Generates a grid of points given by specified number of x and y points
    """
    return np.array([[x, y] for x in np.linspace(0, 1, nx) 
                     for y in np.linspace(0, 1, ny)])   


def full_data_vertical_boundary(n_x, n_y):
    """
    Generates a full grid of points with corresponding labels.
        If the point has x component > 1/2, label is 0. Otherwise label is 1.
        Corresponds to linear vertical decision boundary
    """
    data = grid_pts(n_x, n_y)
    true_labels = np.zeros(n_x*n_y, dtype=int)
    
    for ii in range(data.shape[0]):
        if data[ii][0] < 0.5:
            true_labels[ii] = 1
    return data, true_labels

def full_data_diagonal_boundary(n_x, n_y):
    """
    Generates a full grid of points with corresponding labels.
        If the point has x component > y component, label is 0. Otherwise label is 1.
        Corresponds to linear decision boundary across diagonal of grid.
    """
    data = grid_pts(n_x, n_y)
    true_labels = np.zeros(n_x*n_y, dtype=int)
    
    for ii in range(data.shape[0]):
        if data[ii][0] > data[ii][1]:
            true_labels[ii] = 1
    return data, true_labels

def random_data_vertical_boundary(num_samples, seed=None):
    """Returns a LabeledCData object with randomly sampled data points
    in the (two-dimensional) unit square. Points left of the line
    x = 0.5 are labeled 0, and points right of the line are labeled 1.

    Args:
        num_samples : int
            Number of data points to return.
    """
    # Seed the random number generator if one is provided
    if seed:
        np.random.seed(seed)

    # Get some random data
    data = np.random.rand(num_samples, 2)

    # Do the labeling
    labels = []
    for ii, point in enumerate(data):
        if point[0] > 0.5:
            labels.append(0)
        else:
            labels.append(1)

    return LabeledCData(data, labels)




def random_data_diagonal_boundary(num_samples, seed=None):
    """Returns a LabeledCData object with randomly sampled data points
    in the (two-dimensional) unit square. Points in the lower right diagonal of 
    the unit square are labeled 0, points in the upper left diagonal are labelled 1

    Args:
        num_samples : int
            Number of data points to return.
    """
    # Seed the random number generator if one is provided
    if seed:
        np.random.seed(seed)

    # Get some random data
    data = np.random.rand(num_samples, 2)

    # Do the labeling
    labels = []
    for point in data:
        if point[1] <  point[0] :
            labels.append(0)
        else:
            labels.append(1)

    return LabeledCData(data, labels)

### Generate data set of choice:
def generate_data(data_choice, num_points, random_seed=None, split=False, split_ratio=0.8):
    """
    Generates a dataset with corresponding labels
        Can generate:
            1) Random points with linear vertical decision boundary
            2) Random points with linear diagonal decision boundary
            3) Full grid of data points with linear vertical decision boundary
            4) Full grid of data points with linear diagonal decision boundary
        Args:
            data_choice: str
                choice of dataset to generate
            num_points : int
                Number of datapoints to generate
            random_seed : int
                fix random seed as an option
            split : bool
                if True, split data into training and test, otherwise don't
            split_ratio : float
                if split==True, split data according to this ratio. 

    """
    if split:
        num_train_points = num_points * split_ratio
        num_test_points = num_points * (1-split_ratio)
        if data_choice.lower() == 'random_vertical_boundary':
            cdata_train = random_data_vertical_boundary(num_train_points, seed=random_seed)
            cdata_test = random_data_vertical_boundary(num_test_points, seed=random_seed)
        elif data_choice.lower() == 'random_diagonal_boundary':
            cdata_train = random_data_diagonal_boundary(num_train_points, seed=random_seed)
            cdata_test = random_data_diagonal_boundary(num_test_points, seed=random_seed)

        train_data = cdata_train.data
        train_true_labels = cdata_train.labels
        test_data = cdata_train.data
        test_true_labels = cdata_test.labels

        return train_data, test_data, train_true_labels, test_true_labels
    else:
        if data_choice.lower() == 'random_vertical_boundary':
            cdata = random_data_vertical_boundary(num_points, seed=random_seed)
            data = cdata.data
            true_labels = cdata.labels
        elif data_choice.lower() == 'random_diagonal_boundary':
            cdata = random_data_diagonal_boundary(num_points, seed=random_seed)
            data = cdata.data
            true_labels = cdata.labels

        elif data_choice.lower() == 'full_vertical_boundary':
            data, true_labels = full_data_vertical_boundary(int(np.sqrt(num_points)),int(np.sqrt(num_points)))

        elif data_choice.lower() == 'full_diagonal_boundary':
            data, true_labels = full_data_diagonal_boundary(int(np.sqrt(num_points)), int(np.sqrt(num_points)))
    
        return data, true_labels