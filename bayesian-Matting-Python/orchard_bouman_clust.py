import numpy as np

# Implementation of orchard bouman clustering in 1991 by Orchard and Bouman

class Node(object):

    '''
    Input:
    matrix: a numpy array of size nxd, representing a set of n d-dimensional data points.
    w: a numpy array of size n, representing the weight for each data point in matrix.
    
    Output:
    An instance of the Node class.

    Basic Function:
    Initializes a Node object with the given data points and weights.
    Calculates the mean and covariance of the data points using the given weights.
    Calculates the largest eigenvalue and corresponding eigenvector of the covariance matrix.
    Stores the calculated values as attributes of the Node object.
    '''

    def __init__(self, matrix, w):
        W = np.sum(w)
        self.w = w
        self.X = matrix
        self.left = None
        self.right = None
        self.mu = np.einsum('ij,i->j', self.X, w)/W
        diff = self.X - np.tile(self.mu, [np.shape(self.X)[0], 1])
        t = np.einsum('ij,i->ij', diff, np.sqrt(w))
        self.cov = (t.T @ t)/W + 1e-5*np.eye(3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.cov)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]



def clustFunc(S, w, minVar=0.05):
    '''
    Input:
    S: a numpy array representing the measurements vector, with shape (n, d).
    w: a numpy array representing the weights vector, with shape (n,).
    minVar: a float representing the minimum variance allowed for splitting the nodes. Default value is 0.05.

    Output:
    mu: a numpy array containing the mean vectors of the clusters, with shape (k, d), where k is the number of clusters.
    sigma: a numpy array containing the covariance matrices of the clusters, with shape (k, d, d).

    Basic Function:
    This function performs clustering of the input data using a hierarchical approach based on the binary splitting of nodes. 
    The measurements vector S and the weights vector w are used to initialize a root node, which is then recursively split until no node has variance 
    greater than the specified minimum variance minVar. The mean vectors and covariance matrices of the resulting clusters are then computed and 
    returned as output.
    '''

    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))

    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.cov)

    return np.array(mu), np.array(sigma)


def split(nodes):
    '''
    Input:
    nodes: a list of Node objects

    Output:
    nodes: a list of Node objects after splitting the node with the highest value of lambda
    
    Basic Function:
    This function takes a list of Node objects as input and identifies the node with the highest value of lambda. The node is split into two separate nodes
    based on a condition where the dot product of the node's data with its eigenvector is compared to the dot product of the node's mean with its eigenvector. 
    Two new nodes are created and added to the list, while the original node is removed. The function returns the updated list of nodes.
    '''
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[np.logical_not(idx)], C_i.w[np.logical_not(idx)])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes
