import matplotlib.pyplot as plt

from plots import scatter
from data_gen import generate_data

"""
    This function plots correctly labelled datapoints from a particular 2D dataset:
    moons, random_vertical_boundary, random_diagonal_boundary
"""
def main(data_choice='random_vertical_boundary'):

    # For Moons dataset, noise = 0.05 is added.

    data_train, data_test, true_labels_train, true_labels_test = generate_data(data_choice, num_points=1024, split=True)
    true_labels_test_flip = []
    for label in true_labels_test:
        if label == 0: true_labels_test_flip.append( 1 )
        elif label == 1: true_labels_test_flip.append( 0 ) 
        else: raise ValueError('This label does not exist')

    plot_params_train = {'colors': ['blue', 'magenta'], 'alpha': 0.5, 'size': 80}
    scatter(data_train, true_labels_train, **plot_params_train, show=False)
    plot_params_test =  {'colors': ['blue', 'magenta'], 'alpha': 1, 'size': 40, 'linewidth' : 1.5}
    scatter(data_test, true_labels_test, true_labels_test_flip, **plot_params_test, show=False)
    plt.show()


if __name__ == "__main__":
    main(data_choice='moons')
    main(data_choice='random_vertical_boundary')
    main(data_choice='random_diagonal_boundary')
