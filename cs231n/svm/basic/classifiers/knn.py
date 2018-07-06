import numpy as np

class KNearestNeighbour:
    """L2 distance"""


    def __init__(self):
        pass
    def train(self, x, y):
        """
        train the classifier
        inputs:
        x:num_train x dimension array whiere each row is a training point
        y:a vector of length num_train ,where y[i] is the label for x[i, :]
        """

        self.x_train = x
        self.y_train = y


    def predict(self, x, k=1, num_loops=0):
        """
        predict
        input:
        x:num_test x dimension array whiere each row is a test point
        """
        if num_loops == 0:
            dists = self.compute_distance_no_loops(x)
        elif num_loops == 1:
            dists = self.compute_distance_one_loop(x)
        elif num_loops == 2:
            dists = self.compute_distance_two_loop(x)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists,k=k)

    def compute_distance_two_loops(self, x):
        """
        compute the L2ditance by using the 2 for loops
        input:
        :param x:
        :return:
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros(num_test, num_train)
        for i in xrange(num_test):
            for j in xrange(num_train):
                #distance between test image of i and training image of j
                dists[i, j] = np.sum(self.x_train[j, :] - x[i, :] ** 2)
        return dists
    def compute_distance_one_loop(self, x):
        """
        compute the distance between each test point in x and each training point in self.x_train using a single for loop
        over the test data
        :param x:
        :return:
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros(num_test, num_train)
        for i in xrange(num_test):
            train_2 = np.sum((self.x_train) ** 2, axis=1).T
            test_2 = np.tile(np.sum(x[i, :] ** 2), [1, num_train])
            test_train = x[i, :].dot(self.x_train.T)
            dists[i, :] = train_2 + test_2 - 2 * test_train
        return dists
    def compute_dsitance_no_loop(self, x):
        """
        no loop
        :param x:
        :return:
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros(num_test, num_train)
        #expand the fomula and calculate each term respectively
        train_2 = np.tile(np.sum((self.x_train)**2, axis=1), [num_test, 1])
        test_2 = np.tile(np.sum(x ** 2, axis=1), [num_train, 1]).T
        test_train = x.dot(self.x_train.T)
        dists = train_2 + test_2 - 2 * test_train
        return dists
    def predict_labels(self, dists, k=1):
        """
        discriminate the class,giving the distance
        :param dists:
        :param k:
        :return:
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            #a list of length k storing the labels of the k nearest neighbor to the ith test point
            closest_y = []
            closest_idx = np.argsort(dists[i, :])[: k].tolist()
            closest_y = self.y_train[closest_idx]
            #count the frequency of those closest labels
            counts = np.bincount(closest_y)
            #return the most frequent item as result
            y_pred[i] = np.argmax(counts)
        return y_pred

















































































































