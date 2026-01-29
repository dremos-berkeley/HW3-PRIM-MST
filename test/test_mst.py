import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # Check that MST has correct number of edges (n-1 for n vertices)                              
    n = mst.shape[0]                                                                               
    num_edges = np.sum(mst > 0) // 2  # divide by 2 since matrix is symmetric                      
    assert num_edges == n - 1, f'MST should have {n-1} edges, but has {num_edges}'                 
                                                                                                    
    # Check that MST is symmetric (undirected)                                                     
    assert np.allclose(mst, mst.T), 'MST adjacency matrix should be symmetric'                     
                                                                                                    
    # Check that all MST edges exist in original graph                                             
    for i in range(n):                                                                             
        for j in range(i + 1, n):                                                                  
            if mst[i, j] > 0:                                                                      
                assert adj_mat[i, j] > 0, f'MST contains edge ({i},{j}) not in original graph'

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """                                                                   
    adj_mat = np.array([                                                                           
        [0, 2, 3],                                                                                 
        [2, 0, 1],                                                                                 
        [3, 1, 0]                                                                                  
    ], dtype=float)                                                                                
                                                                                                    
    g = Graph(adj_mat)                                                                             
    g.construct_mst()                                                                              
                                                                                                    
    check_mst(g.adj_mat, g.mst, expected_weight=3)
