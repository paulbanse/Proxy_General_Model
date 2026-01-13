import numpy as np

class EchoStateNetwork:
    def __init__(self, n, spectral_radius=0.9, sparsity=0.1, alpha = 0.5):
        """
        Initialize the Echo State Network (ESN).

        Parameters
        ----------
        n : int
            Number of nodes in the reservoir.
        spectral_radius : float, optional
            Scaling factor to adjust the largest eigenvalue of the connectivity matrix, by default 0.9.
        sparsity : float, optional
            Probability of each connection being nonzero, by default 0.1.
        alpha : float, between 0 and 1, optional
            low pass filter parameter, lower value increase oscillation periods
        """
        self.n = n
        self.W = self._initialize_reservoir(n, spectral_radius, sparsity)
        self.connectivity = np.where(self.W !=0, 1, 0)
        self.state = np.zeros(n)
        self.alpha = alpha

    def __init__(self, n, spectral_radius=0.9, avg_number_of_edges=2, action_nodes=[1], alpha = 0.5):
        """
        Initialize the Echo State Network (ESN) ensuring full percolation of action nodes.
        This version of the ESN ensures that all action nodes are connected to each other, allowing for full percolation.

        Parameters
        ----------
        n : int
            Number of nodes in the reservoir.
        spectral_radius : float, optional
            Scaling factor to adjust the largest eigenvalue of the connectivity matrix, by default 0.9.
        avg_number_of_edges : float, optional
            Average number of edges per node, by default 2.
        action_nodes : list, optional
            List of nodes that should be connected to each other, by default [1].
        alpha : float, between 0 and 1, optional
            Low pass filter parameter, lower value increases oscillation periods.
        """
        self.n = n
        self.W = self._initialize_useful_reservoir(n, spectral_radius, avg_number_of_edges, action_nodes)
        self.connectivity = np.where(self.W !=0, 1, 0)
        self.state = np.zeros(n)
        self.alpha = alpha

    def _initialize_reservoir(self, n, spectral_radius, sparsity):
        """
        Initialize the reservoir connectivity matrix.

        Parameters
        ----------
        n : int
            Number of nodes in the reservoir.
        spectral_radius : float
            Scaling factor to adjust the largest eigenvalue of the connectivity matrix.
        sparsity : float
            Probability of each connection being nonzero.

        Returns
        -------
        np.ndarray
            Initialized connectivity matrix.
        """
        W = np.random.rand(n, n) - 0.5
        W *= (np.random.rand(n, n) < sparsity)  # Apply sparsity mask
        max_eigval = np.max(np.abs(np.linalg.eigvals(W)))
        return W * (spectral_radius / max_eigval) if max_eigval > 0 else W
    
    def _initialize_useful_reservoir(self, n, spectral_radius, avg_number_of_edges, action_nodes):
        """
        Initialize the reservoir connectivity matrix with a specific structure.
        Parameters
        ----------
        n : int
            Number of nodes in the reservoir.
        spectral_radius : float
            Scaling factor to adjust the largest eigenvalue of the connectivity matrix.
        avg_number_of_edges : float
            Average number of edges per node.
        action_nodes : list
            List of nodes that should be connected to each other.
        Returns
        -------
        np.ndarray
            Initialized connectivity matrix with specific structure.
        """
        W = np.zeros((n, n))
        unconnected_nodes = [i for i in range(n) if i not in action_nodes]
        connected_nodes = list(action_nodes)
        while unconnected_nodes:
            # Randomly select an action node to connect to
            connect_from = np.random.choice(unconnected_nodes)
            connect_to = np.random.choice(connected_nodes)
            W[connect_from, connect_to] = np.random.rand()-0.5
            unconnected_nodes.remove(connect_from)
            connected_nodes.append(connect_from)
        
        total_posssible_edges = n * (n - 1) - n + len(action_nodes)
        probability_of_edge = (avg_number_of_edges-1)*n / total_posssible_edges

        for k in range(n):
            for j in range(n):
                if np.random.rand() < probability_of_edge and W[k, j] == 0 and k != j:
                    W[k, j] = np.random.rand() - 0.5
        max_eigval = np.max(np.abs(np.linalg.eigvals(W)))
        return W * (spectral_radius / max_eigval) if max_eigval > 0 else W

    def step(self, agent_nodes, agent_values):
        """
        Update the reservoir state for one time step.

        Parameters
        ----------
        agent_nodes : array-like
            Indices of nodes controlled by the agent.
        agent_values : array-like
            Fixed values (-1, 0, 1) assigned to the agent-controlled nodes.

        Returns
        -------
        np.ndarray
            Updated reservoir state.
        """
        new_state = (1-self.alpha) * self.state + self.alpha* np.tanh(self.W @ self.state)
        self.state = new_state
        print(agent_nodes,new_state[agent_nodes], agent_values)
        
        return self.state

    def run(self, steps, discard=100, agent_nodes=None, agent_values=None):
        """
        Run the ESN for a given number of steps with agent control and return the states over time.

        Parameters
        ----------
        steps : int
            Total number of time steps to run.
        discard : int, optional
            Number of initial steps to discard (transient period), by default 100.
        agent_nodes : array-like, optional
            Indices of nodes controlled by the agent, by default None.
        agent_values : array-like, optional
            Fixed values (-1, 0, 1) assigned to agent-controlled nodes, by default None.

        Returns
        -------
        np.ndarray
            Array of reservoir states over time.
        """
        collected_states = []
        self.state = np.zeros(self.n)
        if agent_nodes is None:
            agent_nodes = []
        if agent_values is None:
            agent_values = np.zeros(len(agent_nodes))
        
        self.state[agent_nodes] = agent_values  # Override controlled nodes

        for t in range(steps):
            #self.state[agent_nodes] = agent_values
            if t >= discard:
                collected_states.append(self.state.copy())
            self.step(agent_nodes, np.zeros(len(agent_nodes)))
        
        return np.array(collected_states)
    
    def distance(self, nodes1, nodes2):
        """
        Compute the distance between objective and proxy nodes.

        Parameters
        ----------
        states : np.ndarray
            Array of reservoir states over time.
        obj_nodes : array-like
            Indices of objective nodes.
        proxy_nodes : array-like
            Indices of proxy nodes.

        Returns
        -------
        np.ndarray
            Array of distances between objective and proxy nodes.
        """
        self.state = np.zeros(self.n)
        self.state[nodes1] = 1
        for k in range(self.n):
            if any(self.state[nodes2]):
                return k
            self.state = np.where(self.connectivity @ self.state !=0, 1, 0) 
        
        return self.n+1


def compute_averages(states, obj_nodes, proxy_nodes):
    """
    Compute time-averaged values for objective and proxy nodes.

    Parameters
    ----------
    states : np.ndarray
        Array of reservoir states over time.
    obj_nodes : array-like
        Indices of objective nodes.
    proxy_nodes : array-like
        Indices of proxy nodes.

    Returns
    -------
    tuple
        Averages of objective and proxy nodes.
    """
    avg_obj = np.mean(states[:, obj_nodes])
    avg_proxy = np.mean(states[:, proxy_nodes])
    return avg_obj, avg_proxy
