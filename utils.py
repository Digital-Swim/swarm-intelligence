import numpy as np


def indeces_of_top_n_elements(array, n):
    """
    Returns the indices of the top n elements in a 1D array.
    
    Parameters:
    - array: A 1D numpy array.
    - n: The number of top elements to find.
    
    Returns:
    - A list of indices corresponding to the top n elements in the array.
    """
    if n <= 0:
        return []
    
    # Use np.argpartition for efficient selection of top n elements
    partitioned_indices = np.argpartition(array, -n)[-n:]
    
    # Sort the selected indices based on their values in descending order
    sorted_indices = partitioned_indices[np.argsort(array[partitioned_indices])[::-1]]
    
    return sorted_indices.tolist()



###### TEST CASES ######


# Test cases for the indeces_of_top_n_elements function
def test_indeces_of_top_n_elements():
    array = np.array([10, 20, 30, 40, 50])
    n = 3
    expected_indices = [4, 3, 2]  # Indices of the top 3 elements (50, 40, 30)
    result = indeces_of_top_n_elements(array, n)
    assert result == expected_indices, f"Expected {expected_indices}, but got {result}"
    
    n = 0
    expected_indices = []
    result = indeces_of_top_n_elements(array, n)
    assert result == expected_indices, f"Expected {expected_indices}, but got {result}"
    
    n = -1
    expected_indices = []
    result = indeces_of_top_n_elements(array, n)
    assert result == expected_indices, f"Expected {expected_indices}, but got {result}"
    
    print("All tests passed!")
    
if __name__ == "__main__":
    test_indeces_of_top_n_elements()