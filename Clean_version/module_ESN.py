import numpy as np

class EchoStateNetwork:

    def __init__(self,
                n,
                spectral_radius=0.9, 
                avg_number_of_edges=2,
                alpha = 0.1, 
                proxy_discard = 50,
                goal_discard = 100, 
                measure_time = 50, 
                seed = None, 
                number_of_action_nodes =16):
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
        self.goal = [n-1]
        assert(n> number_of_action_nodes + 2)
        self.number_of_action_nodes = number_of_action_nodes
        self.action_nodes = [k for k in range(number_of_action_nodes)]
        self.spectral_radius = spectral_radius
        self.avg_number_of_edges = avg_number_of_edges
        if seed is None:
            seed = np.random.randint(0, 1000000)
        self.seed = seed
        np.random.seed(seed)
        self.W = self._initialize_useful_reservoir()
        self.connectivity = np.where(self.W !=0, 1, 0)
        self.state = np.zeros(n)
        self.alpha = alpha
        self.proxy_discard = proxy_discard
        self.goal_discard = goal_discard
        self.measure_time = measure_time
        self.endstep = max(self.proxy_discard, self.goal_discard)+ self.measure_time


    
    def _initialize_useful_reservoir(self):
        """
        Initialize the reservoir connectivity matrix with a specific structure.
        Parameters
        -------
        np.ndarray
            Initialized connectivity matrix with specific structure.
        """
        W = np.zeros((self.n, self.n))
        unconnected_nodes = [i for i in range(self.n) if i not in self.action_nodes]
        connected_nodes = list(self.action_nodes)
        while unconnected_nodes:
            # Randomly select an action node to connect to
            connect_to = np.random.choice(unconnected_nodes)
            connect_from = np.random.choice(connected_nodes)
            W[connect_to,connect_from] = np.random.rand()-0.5
            unconnected_nodes.remove(connect_to)
            connected_nodes.append(connect_to)
        
        total_posssible_edges = self.n * (self.n - 1) - self.n + len(self.action_nodes)
        probability_of_edge = (self.avg_number_of_edges-1)*self.n / total_posssible_edges

        for k in range(self.n):
            for j in range(self.n):
                if np.random.rand() < probability_of_edge and W[k, j] == 0 and k != j:
                    W[k, j] = np.random.rand() - 0.5
        max_eigval = np.max(np.abs(np.linalg.eigvals(W)))
        return W * (self.spectral_radius / max_eigval) if max_eigval > 0 else W

    def step(self):
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
        
        return self.state

    def run(self, agent_values):

        collected_states = []
        self.state = np.zeros(self.n)
        self.state[self.action_nodes ] = agent_values  # Override controlled nodes

        for t in range(self.endstep):
            collected_states.append(self.state.copy())
            self.step()
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
    
    def reset_seed(self):
        np.random.seed(self.seed)


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

def get_base_action_value(size = None):
    """
    Compute the action values if all actions are neutral according to a beta distribution between -1 and 1 of parameter 2 and 2
    """
    a, b = (2,2)
    return 2*np.random.beta( a, b, size)-1

def get_directed_action_value(action_value , target_node, actionsize = None):
    """
    Compute the action values if one action is targetted to be enforced (+1) or reduced (-1) 
    """
    params = [(2,2),(20,2),(2,20)]
    A_vals = get_base_action_value(actionsize)
    a,b = params[action_value]
    A_vals[target_node]= 2*np.random.beta( a, b)-1
    return A_vals