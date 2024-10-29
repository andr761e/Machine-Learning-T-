import numpy as np
from sklearn.model_selection import train_test_split
import pdb 
from sklearn.tree import DecisionTreeRegressor
from io import StringIO  
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus

from sklearn.datasets import fetch_california_housing

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

def plot_tree(dtree, feature_names):
    """ helper function """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print('exporting tree to dtree.png')
    graph.write_png('dtree.png')


class RegressionStump():
    
    def __init__(self):
        """ The state variables of a stump"""
        self.idx = None
        self.val = None
        self.left = None
        self.right = None
    
    def fit(self, data, targets):
        """ Fit a decision stump to data
        
        Find the best way to split the data, minimizing the least squares loss of the tree after the split 
    
        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets
    
        sets self.idx, self.val, self.left, self.right
        """
        # update these three
        self.idx = 0
        self.val = None
        self.left = None
        self.right = None

        ### YOUR CODE HERE
        n, d = data.shape
        best_loss = float('inf')
        print(n)
        print(d)

        #Iterate over each feature
        for i in range(d):
            unique_values = np.unique(data[:, i])
            for v in unique_values:
                left_mask = data[:, i] < v
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                left_mean = targets[left_mask].mean()
                right_mean = targets[right_mask].mean()

                # Compute the loss
                left_loss = np.sum((targets[left_mask] - left_mean) ** 2)
                right_loss = np.sum((targets[right_mask] - right_mean) ** 2)
                loss = left_loss + right_loss

                #Update the stump if this is the best split so far
                if loss < best_loss:
                    best_loss = loss
                    self.idx = i
                    self.val = v
                    self.left = left_mean
                    self.right = right_mean
        ### END CODE

    def predict(self, X):
        """ Regression tree prediction algorithm

        Args
            X: np.array, shape n,d
        
        returns pred: np.array shape n,  model prediction on X
        """
        pred = None
        ### YOUR CODE HERE
        "Uses the trained stump to predict the value for each data point." 
        "If the data point's feature value is less than the split value, it returns the left leafs value; otherwise, it returns the right leafs value"
        pred = np.where(X[:, self.idx] < self.val, self.left, self.right)
        ### END CODE
        return pred
    
    def score(self, X, y):
        """ Compute mean least squares loss of the model

        Args
            X: np.array, shape n,d
            y: np.array, shape n, 

        returns out: scalar - mean least squares loss.
        """
        out = None
        ### YOUR CODE HERE
        predictions = self.predict(X)
        out = np.mean((y - predictions) ** 2)
        ### END CODE
        return out
        


def main():
    """ Simple method testing """
    housing = fetch_california_housing()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(housing.data,
                                                        housing.target,
                                                        test_size=0.2)

    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Lets see if we can do better with just one question')
    D = RegressionStump()
    D.fit(X_train, y_train)
    print('idx, val, left, right', D.idx, D.val, D.left, D.right)
    print('Feature name of idx', housing.feature_names[D.idx])
    print('Score of model', D.score(X_test, y_test))
    print('lets compare with sklearn decision tree')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names - for comparison', list(enumerate(housing.feature_names)))
    plot_tree(dc, housing.feature_names)

if __name__ == '__main__':
    main()

