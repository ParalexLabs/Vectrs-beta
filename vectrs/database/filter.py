import numpy as np

def apply_filters(vectors, filters):
    """
    Apply filters to a list of vectors.

    Parameters:
        vectors (list of tuples): List of tuples, each containing an ID and a vector, e.g., [(id1, vector1), (id2, vector2), ...]
        filters (dict): Filter criteria, e.g., {'min_norm': 0.5, 'max_norm': 1.5}

    Returns:
        list: Filtered list of vector tuples.
    """
    filtered_vectors = []
    for vector_id, vector in vectors:
        norm = np.linalg.norm(vector)
        if (filters.get('min_norm') is None or norm >= filters['min_norm']) and \
           (filters.get('max_norm') is None or norm <= filters['max_norm']):
            filtered_vectors.append((vector_id, vector))
    return filtered_vectors

def filter_by_id(vectors, target_ids):
    """
    Filter vectors by specific IDs.

    Parameters:
        vectors (list of tuples): List of tuples, each containing an ID and a vector.
        target_ids (set): Set of IDs to keep.

    Returns:
        list: Filtered list of vector tuples.
    """
    return [vector for vector in vectors if vector[0] in target_ids]

def apply_complex_filters(vectors, criteria):
    """
    Apply a complex set of filters to a list of vectors based on multiple criteria.

    Parameters:
        vectors (list of tuples): Vectors to filter.
        criteria (dict): Complex filtering criteria.

    Returns:
        list: Filtered list of vectors.
    """
    filtered_vectors = []
    for vector_id, vector in vectors:
        include_vector = True
        for key, value in criteria.items():
            if key == 'norm_range' and not (value[0] <= np.linalg.norm(vector) <= value[1]):
                include_vector = False
            elif key == 'dot_product' and np.dot(vector, value['vector']) < value['threshold']:
                include_vector = False
            # Add more custom filtering logic here
        if include_vector:
            filtered_vectors.append((vector_id, vector))
    return filtered_vectors

def apply_filters_efficiently(vectors, min_norm=None, max_norm=None):
    vector_ids, vector_array = zip(*vectors)  # Unzip into separate lists
    vector_array = np.array(vector_array)  # Convert list of vectors to a numpy array for efficient processing
    norms = np.linalg.norm(vector_array, axis=1)  # Calculate norms across the array
    condition = np.ones_like(norms, dtype=bool)  # Initialize all True for inclusion
    if min_norm is not None:
        condition &= (norms >= min_norm)
    if max_norm is not None:
        condition &= (norms <= max_norm)
    filtered_vectors = [(vector_ids[i], vector_array[i]) for i in range(len(vector_ids)) if condition[i]]
    return filtered_vectors
