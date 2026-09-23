"""Linear-algebra and graph utilities shared by the summarizers."""
import numpy as np
import heapq

def dijkstra(dist_matrix, start, end, min_dist=0, max_dist=np.inf):
    """
    Dijkstra's algorithm to find the shortest path in a graph.
    :param dist_matrix: 2D numpy array representing the distance matrix of the graph
    :param start: Starting node index
    :param end: Ending node index
    """
    num_nodes = dist_matrix.shape[0]
    visited = np.full(num_nodes, False)
    dist = np.full(num_nodes, np.inf)
    prev = np.full(num_nodes, -1)

    dist[start] = 0
    heap = [(0, start)]

    while heap:
        current_dist, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        if u == end:
            break

        for v in range(num_nodes):
            if not visited[v] and dist_matrix[u][v] != min_dist and dist_matrix[u][v] != max_dist:
                alt = current_dist + dist_matrix[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

    if dist[end] == np.inf:
        return []

    # Reconstruct path
    path = []
    u = end
    while u != -1:
        path.append(u)
        u = prev[u]
    return path[::-1]

def cos_sim_mat(tf_matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    col_norms = np.clip(np.linalg.norm(tf_matrix, axis=0), 1, None)
    sim_matrix = np.dot(tf_matrix.T, tf_matrix) / np.outer(col_norms, col_norms)
    sim_matrix[sim_matrix < threshold] = 0
    return sim_matrix

def overlap_sim_mat(tf_matrix: np.ndarray) -> np.ndarray:
    log_sent_len = np.clip(np.log(np.sum(tf_matrix, axis=0) + 1), 1e-100, None)
    sim_matrix = np.dot(tf_matrix.T, tf_matrix)/np.add.outer(log_sent_len, log_sent_len)
    return sim_matrix

def markov_chain(transition_matrix: np.ndarray, v: np.ndarray = None, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    num_nodes = transition_matrix.shape[0]
    if v is None:
        v = np.ones(num_nodes) / num_nodes
    else:
        v = v
    for _ in range(max_iter):
        v_next = np.dot(transition_matrix.T, v)
        if np.linalg.norm(v - v_next) < tol:
            break
        v = v_next
    return v

def markov_cluster(transition_matrix: np.ndarray, expansion: int = 2, inflation: float = 2.0, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    # Normalize the input matrix to ensure it's a stochastic matrix (rows sum to 1)
    for _ in range(max_iter):
        prev_matrix = transition_matrix.copy()

        # Expansion step -- controls the speed of the clusters i.e. higher values = global clusters 
        transition_matrix = np.linalg.matrix_power(transition_matrix, expansion)

        # Inflation step (element-wise power + normalization) -- controls how softly the clusters are defined
        transition_matrix = np.power(transition_matrix, inflation)
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        # Check for convergence
        if np.allclose(transition_matrix, prev_matrix, atol=tol):
            break

    return transition_matrix

def louvain_numpy(similarity_matrix: np.ndarray, max_iter=100):
    A = similarity_matrix.copy()
    n = A.shape[0]
    communities = np.arange(n)  # Start: each node is its own community

    def modularity(A, communities):
        m = np.sum(A)
        Q = 0.0
        degrees = np.sum(A, axis=1)
        for i in range(n):
            for j in range(n):
                if communities[i] == communities[j]:
                    Q += A[i][j] - (degrees[i] * degrees[j]) / m
        return Q / m

    def best_community_for_node(i, A, communities, m, degrees):
        best_comm = communities[i]
        best_gain = 0.0
        current_comm = communities[i]

        # Try all existing communities
        for comm in np.unique(communities):
            if comm == current_comm:
                continue

            in_comm = (communities == comm)
            ki_in = np.sum(A[i][in_comm])
            sum_tot = np.sum(degrees[in_comm])
            ki = degrees[i]
            delta_q = (ki_in - (ki * sum_tot) / m)

            if delta_q > best_gain:
                best_gain = delta_q
                best_comm = comm

        return best_comm, best_gain

    for it in range(max_iter):
        improvement = False
        m = np.sum(A)
        degrees = np.sum(A, axis=1)

        for i in range(n):
            best_comm, gain = best_community_for_node(i, A, communities, m, degrees)
            if best_comm != communities[i]:
                communities[i] = best_comm
                improvement = True

        if not improvement:
            break

        # Phase 2: Aggregate communities
        unique_comms = np.unique(communities)
        comm_map = {old: new for new, old in enumerate(unique_comms)}
        new_n = len(unique_comms)
        new_A = np.zeros((new_n, new_n))

        for i in range(n):
            for j in range(n):
                ci = comm_map[communities[i]]
                cj = comm_map[communities[j]]
                new_A[ci][cj] += A[i][j]

        A = new_A
        n = A.shape[0]
        communities = np.arange(n)

    return communities