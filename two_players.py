import numpy as np

def compute_proxy_nodes(states, goal_nodes, num_proxy_nodes):
    """
    Compute proxy nodes by selecting nodes that maximize temporal correlation with the goal nodes.

    Parameters
    ----------
    states : np.ndarray
        Array of reservoir states over time (shape: [time_steps, num_nodes]).
    goal_nodes : array-like
        Indices of the goal nodes.
    num_proxy_nodes : int
        Number of proxy nodes to select.

    Returns
    -------
    np.ndarray
        Indices of the selected proxy nodes.
    """
    # Compute the average time series of the goal nodes
    goal_time_series = np.mean(states[:, goal_nodes], axis=1)

    # Compute correlation of each node with the goal time series
    correlations = []
    for node in range(states.shape[1]):
        node_time_series = states[:, node]
        correlation = np.corrcoef(goal_time_series, node_time_series)[0, 1]
        correlations.append(correlation)

    # Sort nodes by correlation in descending order and select top nodes
    sorted_indices = np.argsort(correlations)[::-1]
    proxy_nodes = sorted_indices[:num_proxy_nodes]

    return proxy_nodes