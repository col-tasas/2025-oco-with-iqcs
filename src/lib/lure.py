import numpy as np
import control as ctrl
from scipy import linalg


def lti_stack(sys1, sys2):
    """
    Stacks two Linear Time-Invariant (LTI) systems into a single system.

    Returns:
    - A new LTI system that represents the stacked combination of sys1 and sys2.
    """
    
    A1, B1, C1, D1 = ctrl.ssdata(sys1)
    A2, B2, C2, D2 = ctrl.ssdata(sys2)

    if B1.shape[1] != B2.shape[1] or D1.shape[1] != D2.shape[1]:
        raise ValueError('Error in system stacking: number of inputs must be the same for both subsystems!')

    A = linalg.block_diag(A1, A2)
    B = np.vstack((B1, B2))
    C = linalg.block_diag(C1, C2)
    D = np.vstack((D1, D2))

    return ctrl.ss(A, B, C, D, dt=1)


def build_iqc(m, L, p, q, vIQC=True):
    """
    Builds an Integral Quadratic Constraint (IQC) for a given system.

    Parameters:
    - m: Lower bound for the sector condition.
    - L: Upper bound for the sector condition.
    - p: Number of inputs/outputs for the [m, L]-sector nonlinearity.
    - q: Number of inputs/outputs for the [0, infty]-sector nonlinearity.
    - vIQC: determines whether an off-by-one vIQC is incorporated.


    Returns:
    - An LTI system representing the IQC.
    """

    ### IQC for [m, L]
    A_psi = np.zeros((p,p))
    base_block1 = np.asarray([[L], [-m]])
    base_block2 = np.asarray([[-1], [1]])
    block1 = linalg.block_diag(*([base_block1] * p))
    block2 = linalg.block_diag(*([base_block2] * p))
    blockz = np.zeros((block1.shape[0], q))
    D_psi = np.block([[block1, blockz, block2, blockz]]) 
    B_psi = -D_psi[::2, :]
    base_block_C = np.asarray([[1], [0]])
    C_psi = linalg.block_diag(*([base_block_C] * p))

    Psi_mL_sector = ctrl.ss([], [], [], D_psi, dt=1)
    Psi_mL_offby1 = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)
    
    ### IQC for [0, infty]
    A_psi = np.zeros((q,q))
    base_block1 = np.asarray([[1], [0]])
    base_block2 = np.asarray([[0], [1]])
    block1 = linalg.block_diag(*([base_block1] * q))
    block2 = linalg.block_diag(*([base_block2] * q))
    blockz = np.zeros((block1.shape[0], p))
    D_psi = D_psi = np.block([[blockz, block1, blockz, block2]])
    B_psi = -D_psi[::2, :]
    C_psi = linalg.block_diag(*([base_block_C] * q))

    Psi_zInf_sector = ctrl.ss([], [], [], D_psi, dt=1)
    Psi_zInf_offby1 = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1)

    # create stacked IQC
    if vIQC:
        Psi_sector = lti_stack(Psi_mL_sector, Psi_zInf_sector)
        Psi_offby1 = lti_stack(Psi_mL_offby1, Psi_zInf_offby1)
        Psi = lti_stack(Psi_sector, Psi_offby1)
    else:
        Psi = lti_stack(Psi_mL_sector, Psi_zInf_sector)

    return Psi


def build_lure_system(G, m, L, p, q, vIQC=True):
    """
    Builds a Lur'e system representing the interconnection of algorithm and monotone operator.

    Parameters:
    - G: The plant model in state-space form.
    - m: Lower bound for the sector condition.
    - L: Upper bound for the sector condition.
    - p: Number of inputs/outputs for the [m, L]-sector nonlinearity.
    - q: Number of inputs/outputs for the [0, infty]-sector nonlinearity.
    - vIQC: determines whether an off-by-one vIQC is incorporated.

    Returns:
    - A tuple (A, B, C, D, M) representing the overall state-space system and the multiplier matrix M.
    """

    ### IQC filter and multiplier
    Psi = build_iqc(m, L, p, q, vIQC)
    M = np.asarray([[0,1],
                    [1,0]])

    ### iqc_input = [y; u] = [G*u; u] = [G; I]*u => build [G; I]
    n_g = G.ninputs
    G_I   = lti_stack(G, np.eye(n_g))

    ### psi = Psi * iqc_input => build Psi*[G; I]
    G_hat = ctrl.series(G_I, Psi)
    A, B, C, D = ctrl.ssdata(G_hat)

    return A, B, C, D, M