"""Multi-body allosteric cumulants and higher-order dynamics for Z-DynGNN.

Translates the fast Gaussian Boson Sampling (GBS) emulator framework of
Dodd et al. (Science Advances 2026, DOI: 10.1126/sciadv.aef8431) into protein
biophysics. Extracts 3-body, 4-body, and catalytic hexamer cumulants from
regularized submatrix determinants of GNM/ANM covariance matrices in polynomial time.
"""
import itertools
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


def compute_subsystem_logdet(
    cov_matrix: np.ndarray,
    indices: Sequence[int],
    eps: float = 1e-3,
    is_3d: bool = False,
) -> float:
    """Compute regularized log-determinant of a subsystem covariance matrix.

    Following Dodd et al. (2026), thermal/vacuum regularization eps * I is added
    to the submatrix before computing log-determinant, guaranteeing numerical stability
    against ANM/GNM zero modes (rigid body translations/rotations) and collinearity.

    Args:
        cov_matrix: Full covariance matrix (N x N for GNM, or 3N x 3N for ANM).
        indices: 0-based residue indices belonging to the subsystem.
        eps: Regularization parameter (analogous to hbar / 2 in GBS).
        is_3d: If True, indices map to 3D Cartesian coordinates (3i : 3i+3).

    Returns:
        Regularized log-determinant (float).
    """
    if len(indices) == 0:
        return 0.0

    if is_3d:
        coord_indices = []
        for idx in indices:
            coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
        sub_cov = cov_matrix[np.ix_(coord_indices, coord_indices)]
    else:
        sub_cov = cov_matrix[np.ix_(indices, indices)]

    # Symmetrize defensively
    sub_cov = 0.5 * (sub_cov + sub_cov.T)
    k = sub_cov.shape[0]

    # Add Tikhonov / vacuum regularization eps * I
    reg_sub = sub_cov + eps * np.eye(k, dtype=cov_matrix.dtype)

    # Compute logdet via SVD / eigenvalues for robust positive-definiteness
    sign, logdet = np.linalg.slogdet(reg_sub)
    if sign <= 0 or np.isnan(logdet):
        # Fallback to singular values
        s = np.linalg.svd(reg_sub, compute_uv=False)
        logdet = float(np.sum(np.log(np.maximum(s, 1e-12))))

    return float(logdet)


def find_contact_cliques(
    adj_matrix: np.ndarray,
    max_size: int = 3,
) -> List[Tuple[int, ...]]:
    """Extract structural 3-cliques (closed triangles) from the contact graph.

    Restricts multi-body calculations strictly to mutual contact triangles in the
    protein contact network, avoiding the O(N^3) combinatorial explosion.

    Args:
        adj_matrix: Symmetric binary adjacency matrix (N x N).
        max_size: Maximum clique size (currently 3).

    Returns:
        List of sorted integer tuples representing 3-cliques: (i, j, k) with i < j < k.
    """
    n = adj_matrix.shape[0]
    cliques: List[Tuple[int, ...]] = []

    # Fast triangle extraction via neighbor intersection
    # For every edge (i, j) with i < j, find common neighbors k > j
    adj_bool = (adj_matrix > 0)
    for i in range(n):
        neighbors_i = np.where(adj_bool[i])[0]
        neighbors_i = neighbors_i[neighbors_i > i]
        for j in neighbors_i:
            common = np.where(adj_bool[i] & adj_bool[j])[0]
            common = common[common > j]
            for k in common:
                cliques.append((int(i), int(j), int(k)))

    return cliques


def compute_joint_correlator(
    cov_matrix: np.ndarray,
    subset: Sequence[int],
    k_b_t: float = 1.0,
    eps: float = 1e-3,
    is_3d: bool = False,
) -> float:
    """Compute the multi-residue joint correlator c(S) via subsystem determinants.

    Translates Dodd et al. (Science Advances 2026, Eq. 3):
        c(S) = (-1)^|S| * sum_{R subseteq S} (-2)^|R| / sqrt(det[ 1/(k_B T) * (C_R + eps * I) ])

    Args:
        cov_matrix: Covariance matrix (N x N or 3N x 3N).
        subset: Residue index subset S.
        k_b_t: Thermal energy scaling factor.
        eps: Thermal regularization parameter.
        is_3d: Whether cov_matrix is 3D ANM Hessian pseudo-inverse.

    Returns:
        Joint fluctuation correlator c(S).
    """
    s_list = sorted(list(set(indices_int(subset))))
    d = len(s_list)
    if d == 0:
        return 1.0

    total_sum = 0.0

    # Iterate over all non-empty subsets R of S
    for r_size in range(1, d + 1):
        coeff_r = (-2.0) ** r_size
        for r_comb in itertools.combinations(s_list, r_size):
            logdet_r = compute_subsystem_logdet(cov_matrix, r_comb, eps=eps, is_3d=is_3d)
            dim_r = len(r_comb) * (3 if is_3d else 1)
            # Scaling by 1 / (k_B T): logdet(A / c) = logdet(A) - dim * log(c)
            scaled_logdet = logdet_r - dim_r * np.log(max(k_b_t, 1e-6))
            # 1 / sqrt(det) = exp(-0.5 * logdet)
            inv_sqrt_det = np.exp(-0.5 * scaled_logdet)
            total_sum += coeff_r * inv_sqrt_det

    return float(((-1.0) ** d) * total_sum)


def compute_multibody_cumulant(
    cov_matrix: np.ndarray,
    subset: Sequence[int],
    k_b_t: float = 1.0,
    eps: float = 1e-3,
    is_3d: bool = False,
) -> float:
    """Compute the irreducible multi-body cumulant kappa(S) via Möbius partition inversion.

    Translates Dodd et al. (Science Advances 2026, Eq. 5):
        kappa(S) = sum_{pi in P(S)} (-1)^{|pi| - 1} (|pi| - 1)! * prod_{B in pi} c(B)

    For:
        |S| = 1: kappa({i}) = c({i})
        |S| = 2: kappa({i, j}) = c({i, j}) - c({i}) * c({j})
        |S| = 3: kappa({i, j, k}) = c({i, j, k}) - [c({i})c({j,k}) + c({j})c({i,k}) + c({k})c({i,j})]
                                    + 2 * c({i})c({j})c({k})

    Args:
        cov_matrix: Covariance matrix.
        subset: Residue index subset S.
        k_b_t: Thermal energy scaling factor.
        eps: Thermal regularization parameter.
        is_3d: Whether cov_matrix is 3D ANM Hessian pseudo-inverse.

    Returns:
        Irreducible multi-body cumulant kappa(S).
    """
    s_list = sorted(list(set(indices_int(subset))))
    d = len(s_list)

    if d == 0:
        return 0.0
    if d == 1:
        return compute_joint_correlator(cov_matrix, s_list, k_b_t=k_b_t, eps=eps, is_3d=is_3d)

    if d == 2:
        i, j = s_list[0], s_list[1]
        c_ij = compute_joint_correlator(cov_matrix, [i, j], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_i = compute_joint_correlator(cov_matrix, [i], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_j = compute_joint_correlator(cov_matrix, [j], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        return float(c_ij - c_i * c_j)

    if d == 3:
        i, j, k = s_list[0], s_list[1], s_list[2]
        c_ijk = compute_joint_correlator(cov_matrix, [i, j, k], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_ij = compute_joint_correlator(cov_matrix, [i, j], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_ik = compute_joint_correlator(cov_matrix, [i, k], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_jk = compute_joint_correlator(cov_matrix, [j, k], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_i = compute_joint_correlator(cov_matrix, [i], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_j = compute_joint_correlator(cov_matrix, [j], k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        c_k = compute_joint_correlator(cov_matrix, [k], k_b_t=k_b_t, eps=eps, is_3d=is_3d)

        kappa_3 = (
            c_ijk
            - (c_i * c_jk + c_j * c_ik + c_k * c_ij)
            + 2.0 * c_i * c_j * c_k
        )
        return float(kappa_3)

    # General partition solver for |S| >= 4
    return float(_compute_general_cumulant(cov_matrix, s_list, k_b_t=k_b_t, eps=eps, is_3d=is_3d))


def _compute_general_cumulant(
    cov_matrix: np.ndarray,
    subset: List[int],
    k_b_t: float = 1.0,
    eps: float = 1e-3,
    is_3d: bool = False,
) -> float:
    """Evaluate general cumulant formula using all set partitions."""
    partitions = _get_set_partitions(subset)
    cumulant_val = 0.0

    cache: Dict[Tuple[int, ...], float] = {}

    def get_correlator(block: Tuple[int, ...]) -> float:
        if block not in cache:
            cache[block] = compute_joint_correlator(cov_matrix, block, k_b_t=k_b_t, eps=eps, is_3d=is_3d)
        return cache[block]

    for part in partitions:
        num_blocks = len(part)
        sign_factor = ((-1.0) ** (num_blocks - 1)) * float(math.factorial(num_blocks - 1))
        prod_val = 1.0
        for block in part:
            prod_val *= get_correlator(tuple(sorted(block)))
        cumulant_val += sign_factor * prod_val

    return cumulant_val


def _get_set_partitions(collection: List[int]) -> List[List[List[int]]]:
    """Generate all partitions of a small set recursively."""
    if len(collection) == 1:
        return [[[collection[0]]]]

    first = collection[0]
    rest = collection[1:]
    sub_partitions = _get_set_partitions(rest)
    result = []

    for p in sub_partitions:
        # Option 1: add 'first' as a new singleton block
        result.append([[first]] + p)
        # Option 2: add 'first' to an existing block
        for i in range(len(p)):
            new_p = [block[:] for block in p]
            new_p[i].append(first)
            result.append(new_p)

    return result


def extract_catalytic_hexamer_metrics(
    cov_matrix: np.ndarray,
    cat_indices: Sequence[int] = (77, 79, 81, 146, 165, 207),
    eps: float = 1e-3,
    is_3d: bool = False,
) -> Dict[str, float]:
    """Compute generalized fluctuation metrics for the 6-residue catalytic zinc cage.

    Catalytic residues in mature NDM-1 (positions 43-270):
        His120 (idx 77), His122 (idx 79), Asp124 (idx 81) [Zn1 ligands]
        His189 (idx 146), Cys208 (idx 165), His250 (idx 207) [Zn2 ligands]

    Args:
        cov_matrix: Covariance matrix (N x N or 3N x 3N).
        cat_indices: 0-based indices of the 6 catalytic zinc coordinating residues.
        eps: Thermal regularization parameter.
        is_3d: Whether covariance is 3D ANM Hessian pseudo-inverse.

    Returns:
        Dictionary of biophysical hexamer metrics:
            - hexamer_logdet: Generalized coordination volume log det(C_cat + eps * I).
            - hexamer_trace: Total joint mean-square fluctuation sum(diag(C_cat)).
            - hexamer_mean_corr: Mean off-diagonal pairwise correlation across the cage.
            - hexamer_condition: Regularized condition number.
    """
    cat_idx_clean = sorted([int(x) for x in cat_indices])
    logdet = compute_subsystem_logdet(cov_matrix, cat_idx_clean, eps=eps, is_3d=is_3d)

    if is_3d:
        coord_indices = []
        for idx in cat_idx_clean:
            coord_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
        sub_cov = cov_matrix[np.ix_(coord_indices, coord_indices)]
    else:
        sub_cov = cov_matrix[np.ix_(cat_idx_clean, cat_idx_clean)]

    trace_val = float(np.trace(sub_cov))
    reg_sub = sub_cov + eps * np.eye(sub_cov.shape[0])
    s = np.linalg.svd(reg_sub, compute_uv=False)
    cond_val = float(s[0] / max(s[-1], 1e-12))

    # Mean off-diagonal correlation
    diag_sqrt = np.sqrt(np.maximum(np.diag(sub_cov), 1e-12))
    outer_sqrt = np.outer(diag_sqrt, diag_sqrt)
    corr_mat = sub_cov / outer_sqrt
    np.fill_diagonal(corr_mat, 0.0)
    k = corr_mat.shape[0]
    mean_corr = float(np.sum(corr_mat) / max(1, k * (k - 1)))

    return {
        "hexamer_logdet": float(logdet),
        "hexamer_trace": float(trace_val),
        "hexamer_mean_corr": float(mean_corr),
        "hexamer_condition": float(cond_val),
    }


def compute_node_multibody_features(
    n_nodes: int,
    cliques: List[Tuple[int, ...]],
    clique_cumulants: Dict[Tuple[int, ...], float],
    cat_metrics: Optional[Dict[str, float]] = None,
    cat_indices: Sequence[int] = (77, 79, 81, 146, 165, 207),
) -> np.ndarray:
    """Pool multi-body clique cumulants into node-level feature representations.

    Constructs a feature tensor of shape (N, 6):
        col 0: Mean 3-body cumulant of triangles containing node i.
        col 1: Max absolute 3-body cumulant of triangles containing node i.
        col 2: Sum of 3-body cumulants for node i.
        col 3: Triangle clique participation count for node i (normalized).
        col 4: Binary indicator if node i is a member of the catalytic hexamer.
        col 5: Catalytic hexamer coupling factor (logdet scaled if catalytic, 0 otherwise).

    Args:
        n_nodes: Total number of protein residues N.
        cliques: List of 3-cliques (i, j, k).
        clique_cumulants: Dictionary mapping clique tuple -> cumulant float.
        cat_metrics: Optional catalytic hexamer metrics dictionary.
        cat_indices: Indices of the catalytic zinc ligands.

    Returns:
        np.ndarray of shape (n_nodes, 6), dtype float32.
    """
    features = np.zeros((n_nodes, 6), dtype=np.float32)
    cat_set = set(int(x) for x in cat_indices)

    # Accumulate clique values per node
    node_clique_vals: List[List[float]] = [[] for _ in range(n_nodes)]
    for clq in cliques:
        val = clique_cumulants.get(clq, 0.0)
        for node in clq:
            if 0 <= node < n_nodes:
                node_clique_vals[node].append(val)

    hex_logdet = cat_metrics.get("hexamer_logdet", 0.0) if cat_metrics else 0.0

    for i in range(n_nodes):
        vals = node_clique_vals[i]
        if len(vals) > 0:
            features[i, 0] = float(np.mean(vals))
            features[i, 1] = float(np.max(np.abs(vals)))
            features[i, 2] = float(np.sum(vals))
            features[i, 3] = float(len(vals)) / max(1.0, float(len(cliques)))
        else:
            features[i, 0] = 0.0
            features[i, 1] = 0.0
            features[i, 2] = 0.0
            features[i, 3] = 0.0

        if i in cat_set:
            features[i, 4] = 1.0
            features[i, 5] = float(hex_logdet / 10.0)  # Safe scaling
        else:
            features[i, 4] = 0.0
            features[i, 5] = 0.0

    return features


def indices_int(seq: Sequence[Union[int, np.integer]]) -> List[int]:
    """Helper to ensure clean integer list."""
    return [int(x) for x in seq]
