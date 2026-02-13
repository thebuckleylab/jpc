"""
Utilities to compute DMFT quantities. Adapted to multidimensional outputs from 
"Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural 
Networks" (https://openreview.net/forum?id=sipwrPCrIS).
"""


import jax
import jax.numpy as jnp


def get_Delta(all_H, all_G, Kx, y, eta):
    T, P = all_H[0].shape[0], all_H[0].shape[1]
    
    # Ensure y is (P, K)
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    K = y.shape[1]

    NTK = jnp.einsum('ii,jk->ijk', all_G[0], Kx)
    for l in range(len(all_H)-1):
        NTK += jnp.einsum('ijik, ii->ijk', all_H[l], all_G[l+1])
    NTK += jnp.einsum('ijik->ijk', all_H[-1])

    def step(delta_t, ntk_t):
        grad = (eta / P) * (ntk_t @ delta_t)
        next_delta = delta_t - grad
        return next_delta, delta_t

    # carry: (P, K), scanned: (T-1, P, K)
    last_state, trajectory = jax.lax.scan(step, y, NTK[:-1])
    
    # Concatenate to get (T, P, K)
    return jnp.concatenate([trajectory, last_state[jnp.newaxis, ...]], axis=0)


def solve_kernels(Kx, y, depth, eta, gamma, sigma=1.0, T=100, num_steps=10):
    """
    Unified DMFT Solver for Deep Linear/Non-linear Networks.
    Handles y as (P,) or (P, K).
    """
    # 1. Standardize Input Shape
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    P, K = y.shape
    
    # 2. Initialize Kernels
    # H0 is the base input kernel tiled over time: (T, P, T, P)
    H0 = jnp.broadcast_to(Kx[jnp.newaxis, :, jnp.newaxis, :], (T, P, T, P))
    
    all_H = [sigma**(2*(l+1)) * H0 for l in range(depth)]
    all_G = [sigma**(2*(depth-l)) * jnp.ones((T, T)) for l in range(depth)]
    
    # Intermediate response variables: (T, P, T, K) and (T, T, P, K)
    all_A = [jnp.zeros((T, P, T, K)) for _ in range(depth-1)] 
    all_B = [jnp.zeros((T, T, P, K)) for _ in range(depth-1)] 
    
    eta_g = gamma * eta
    I_TP = jnp.einsum('ik,jl->ijkl', jnp.eye(T), jnp.eye(P)) # Identity (T*P, T*P)
    
    for n in range(num_steps):
        Delta = get_Delta(all_H, all_G, Kx, y, eta)
        
        new_H, new_G, new_A, new_B = [], [], [], []
        
        for l in range(depth):
            # Boundary conditions for depth
            Hminus = H0 if l == 0 else all_H[l-1]
            Aminus = jnp.zeros((T, P, T, K)) if l == 0 else all_A[l-1]
            Gplus = jnp.ones((T, T)) if l == depth - 1 else all_G[l+1]
            Bl = jnp.zeros((T, T, P, K)) if l == depth - 1 else all_B[l]
            
            # Causal masking
            Gtril = jnp.tril(Gplus, k=-1)
            H_tril = jnp.einsum('ik,ijkl->ijkl', jnp.tril(jnp.ones((T, T)), k=-1), Hminus)
            
            # 3. Mean Field Terms (Cl and Dl)
            # Cl: (T, P, T, K), Dl: (T, T, P, K)
            Cl = Aminus + (eta_g / P) * jnp.einsum('ijkl,klm->ijkm', H_tril, Delta)
            Dl = Bl + (eta_g / P) * jnp.einsum('ij,jkm->ijkm', Gtril, Delta)
            
            # 4. Update H Kernel logic
            # Contract over T(k) and K(n)
            CD = jnp.einsum('ijkn,klmn->ijlm', Cl, Dl) 
            Diff_CD_inv = jnp.linalg.inv((I_TP - CD).reshape((T*P, T*P))).reshape((T, P, T, P))
            
            dH1 = jnp.einsum('ijkl,mnkl->ijmn', jnp.einsum('ijkl,klmn->ijmn', Diff_CD_inv, Hminus), Diff_CD_inv)
            A_new = jnp.einsum('ijkl,klmn->ijmn', Diff_CD_inv, Cl)
            dH2 = jnp.einsum('ijkn,kl,mpln->ijmp', A_new, Gplus, A_new)
            new_H.append(dH1 + dH2)
            
            # 5. Update G Kernel logic
            # Contract over T, P, and K
            DC = jnp.einsum('ijkn,jkln->il', Dl, Cl)
            Diff_DC_inv = jnp.linalg.inv(jnp.eye(T) - DC)
            
            dG1 = Diff_DC_inv @ Gplus @ Diff_DC_inv.T
            B_minus_new = jnp.einsum('ij,jklm->iklm', Diff_DC_inv, Dl)
            dG2 = jnp.einsum('ijkn,jklp,mlpn->im', B_minus_new, Hminus, B_minus_new)
            new_G.append(dG1 + dG2)
            
            if l < depth - 1: new_A.append(A_new)
            if l > 0: new_B.append(B_minus_new)
                
        all_H, all_G, all_A, all_B = new_H, new_G, new_A, new_B
            
    return all_H, all_G, all_A, all_B
