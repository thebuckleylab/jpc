"""
Utilities to compute DMFT quantities. Adapted to multidimensional outputs from 
"Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural 
Networks" (https://openreview.net/forum?id=sipwrPCrIS).
"""


import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys



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


def solve_Delta(Kx, y, all_Phi, all_G, eta):
    T = all_Phi[0].shape[0]
    P = all_Phi[0].shape[1]
    Delta = np.zeros((T,P))
    Delta[0,:] = y*1.0
    NTK = jnp.einsum('ijil,jl->ijl', all_G[0], Kx)
    depth = len(all_Phi)
    for l in range(depth-1):
        NTK += jnp.einsum('ijil,ijil->ijl', all_G[l+1], all_Phi[l])
    NTK += jnp.einsum('ijil->ijl', all_Phi[depth-1])
    
    for t in range(T-1):
        Delta[t+1,:] = Delta[t,:] - eta / P * NTK[t,:,:] @ Delta[t,:]    
    return Delta


# computes  < phi(h) phi(h) > for h Gaussian w covariance H
def draw_nonlin_samples_init(H, func, key, samples):
    S, V = jnp.linalg.eigh(H)
    S = S*(S>0.0)
    Z = random.normal(key, (H.shape[0], samples))
    SZ = jnp.einsum('i,ij->ij', S**(0.5), Z)
    h = V @ SZ
    phi = func(h)
    return phi @ phi.T / samples

# computes kernels at init
def initialize_kernels_sampling(Kx, depth, T, nonlin_fn, dnonlin_fn, samples = 1000):
    H = Kx * 1.0
    all_Phi = []
    key = random.PRNGKey(0)
    for l in range(depth):
        Phi = draw_nonlin_samples_init(H, nonlin_fn, key, samples)
        all_Phi += [Phi]
        H = Phi * 1.0
        key,_ = random.split(key)

      
    all_G = [draw_nonlin_samples_init(all_Phi[-2], dnonlin_fn, key, samples)  ]
    for l in range(depth-2):
        key,_ = random.split(key)
        all_G.insert(0,  all_G[0] * draw_nonlin_samples_init(all_Phi[-3-l], dnonlin_fn, key, samples) )
    all_G.insert(0, all_G[0] * draw_nonlin_samples_init(H, dnonlin_fn, key, samples))
    
    all_Phi_tp = []
    all_G_tp = []
    
    for l, Phil in enumerate(all_Phi):
        Gl = all_G[l]
        Phitp = np.zeros((T,P,T,P))
        Gtp = np.zeros((T,P,T,P))
        for t in range(T):
            for s in range(T):
                Phitp[t,:,s,:] = Phil
                Gtp[t,:,s,:] = Gl
        all_Phi_tp += [Phitp]
        all_G_tp += [Gtp]
    
    H0 = np.zeros((T,P,T,P))
    for t in range(T):
        for s in range(T):
            H0[t,:,s,:] = Kx
    
    return all_Phi_tp, all_G_tp, H0


def solve_self_consistent_four_vars(Phi_minus, G_plus, A, B, Delta, t, r, nonlin_fn, dnonlin_fn, eta_gam, num_step = 500):
    h = t * 1.0
    z = r * 1.0
    P = Phi_minus.shape[1]
    
    causal_tt = jnp.tril( jnp.ones((T,T)), k=-1 )
    Phi_tril = jnp.einsum('ijkl,ik->ijkl', Phi_minus, causal_tt)
    C = eta_gam / P * jnp.einsum('ijkl,kl->ijkl', Phi_tril, Delta) 
    
    G_tril = jnp.einsum('ijkl,ik->ijkl', G_plus, causal_tt)
    D = eta_gam / P * jnp.einsum('ijkl,kl->ijkl', G_tril, Delta) 
    #update_h = jit( lambda h,z: t + jnp.einsum('ijk,mnjk->imn', z*dnonlin_fn(h), A+C) )
    #update_z = jit( lambda h: r + jnp.einsum('ijk,mnjk->imn', nonlin_fn(h), B + D) )
    for n in range(num_step):
        
        phi = nonlin_fn(h)
        g = z * dnonlin_fn(h)
        
        chi = t + jnp.einsum('ijk,mnjk->imn', g, A) # chi = t + A g
        xi = r + jnp.einsum('ijk,mnjk->imn', phi, B) # xi = r + B^T phi
        #chi = t 
        #xi = r 
        
        
        h_new = chi + jnp.einsum('ijk,mnjk->imn', g, C)
        z_new = xi + jnp.einsum('ijk,mnjk->imn', phi, D)
        #h_new = update_h(h,z)
        #z_new = update_z(h)
        
        loss = ( jnp.sum((h_new-h)**2) + jnp.sum((z_new-z)**2) ) / (jnp.sum(h**2) + jnp.sum(z**2) )
        h = h_new 
        z = z_new
        if loss < 1e-6:
            break        
    return h, z, chi,xi


# computes Jacobians  dh/du, dh/dr, dz/du, dz/dr
# takes in h, z which solve fixed point equations above 
# and calculates < dphi / dr > and < dg / du > for A, B 
def solve_jacobians_batched(Phi_minus, G_plus, A, B, Delta, h, z, nonlin_fn, dnonlin_fn, ddnonlin_fn, eta_gam, num_step = 30):
    P = Phi_minus.shape[1]
    causal_tt = jnp.tril( jnp.ones((T,T)), k=-1 )
    Phi_tril = jnp.einsum('ijkl,ik->ijkl', Phi_minus, causal_tt)
    C = eta_gam / P * jnp.einsum('ijkl,kl->ijkl', Phi_tril, Delta) + A 
    
    G_tril = jnp.einsum('ijkl,ik->ijkl', G_plus, causal_tt)
    D = eta_gam / P * jnp.einsum('ijkl,kl->ijkl', G_tril, Delta) + B
    
    batch_size = 50
    num_batches = int(h.shape[0] / batch_size)
    dphi_dr_avg = jnp.zeros((T,P,T,P))
    dg_du_avg = jnp.zeros((T,P,T,P))
    
    for batch_n in range(num_batches):
        #sys.stdout.write('\r batch = %d' % batch_n)
        hb = h[batch_n*batch_size:(batch_n+1)*batch_size]
        zb = z[batch_n*batch_size:(batch_n+1)*batch_size]
        id_batch = jnp.einsum('i,jklm->ijklm', jnp.ones(batch_size), jnp.einsum('ik,jl->ijkl', jnp.eye(T), jnp.eye(P)))
        dh_du = id_batch # initial guess h ~ u + O(gamma)
        dz_dr = id_batch # initial guess z ~ r + O(gamma)
        dh_dr = jnp.zeros((batch_size, T,P,T,P))
        dz_du = jnp.zeros((batch_size, T,P,T,P))
        dot_phi_b = dnonlin_fn(hb)
        phi_b = nonlin_fn(hb)
        ddot_phi_b = ddnonlin_fn(hb)
        #get_dg=jit(lambda dz, dh: jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dz) + jnp.einsum('ijk,ijklm->ijklm', ddot_phi_b*zb ,dh))
        #get_dphi=jit(lambda dh: jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dh_du))
        
        for n in range(num_step):
            
            
            # g = dot_phi(h) * z,  
            # dg/dr = (ddot_phi(h) * z) * dh/dr + dot_phi(h) * dz/dr
            # dg/du = ddot_phi(h) * z * dh/du + dot_phi(h)* dz/du
            dg_du = jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dz_du) + jnp.einsum('ijk,ijklm->ijklm', ddot_phi_b*zb ,dh_du)
            dg_dr = jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dz_dr) + jnp.einsum('ijk,ijklm->ijklm', ddot_phi_b*zb, dh_dr)
            #dg_du = get_dg(dz_du, dh_du)
            #dg_dr = get_dg(dz_dr, dh_dr)
            
            
            # dh/du = I + C dg_du , dh/dr = C dg/dr
            dh_du = id_batch + jnp.einsum('jklm, ilmno->ijkno', C, dg_du)
            dh_dr_new = jnp.einsum('jklm, ilmno->ijkno', C, dg_dr)
            movement = jnp.mean((dh_dr_new - dh_dr)**2 ) 
            #if n % 20 == 0:
            #    sys.stdout.write('\r movement: %e' % movement)
            dh_dr = dh_dr_new
            
            # dphi/du = dot_phi(h) * dh/du, dphi/dr = dot_phi(h) * dh/dr
            dphi_du = jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dh_du)
            dphi_dr = jnp.einsum('ijk,ijklm->ijklm', dot_phi_b, dh_dr)
            
            # dz/dr = I + D dphi/dr ,  dz/du = D dphi/du
            dz_dr = id_batch + jnp.einsum('jklm, ilmno->ijkno', D, dphi_dr)
            dz_du = jnp.einsum('jklm, ilmno->ijkno', D, dphi_du)
            
            if movement < 1e-20:
                break
        
        dphi_dr_avg += 1/num_batches * jnp.mean(dphi_dr, axis = 0)
        dg_du_avg += 1/num_batches * jnp.mean(dg_du, axis = 0)
    
    return dphi_dr_avg, dg_du_avg


# need MCMC for layer 1, layers 1< l < L and layer L
# for layer 1, chi does not need to be resampled. For layer L, xi does not need to be resample

def DMFT_Theory_Cross_Term(Kx, y, depth=3, T=100, eta = 0.01, gamma=1.0, num_iter=15, samples = 1000, alpha = 0.8, nonlin = 'softplus', beta = 1.0):
    P = Kx.shape[0]
    # initialize kernels
    #if nonlin == 'tanh':
    #    all_Phi, all_G, H0 = initialize_kernels_tanh(Kx, depth, T)
    #else:
    #    all_Phi, all_G, H0 = initialize_kernels_double(Kx, depth, T)
    #print(len(all_Phi))
    #print(len(all_G))
    #print(all_Phi[0].shape)
    #print(all_G[0].shape)
    
    key = random.PRNGKey(0)
    if nonlin == 'tanh':
        nonlin_fn = lambda h: jnp.tanh(h * beta) / beta
        dnonlin_fn = lambda h: 1.0 - jnp.tanh(h * beta)**2
        ddnonlin_fn = lambda h: 2*beta*jnp.tanh(h*beta)*(jnp.tanh(beta*h)**2 - 1)
    
    else:
        nonlin_fn = lambda h: jnp.sqrt(2.0) * 1/beta*jnp.log(1.0 + jnp.exp(beta*h))
        dnonlin_fn = lambda h: jnp.sqrt(2.0)/(1.0 + jnp.exp(-beta*h))
        ddnonlin_fn = lambda h: jnp.sqrt(2.0)* beta * jnp.exp(-beta*h)/(1.0 + jnp.exp(-beta*h))**2
    
    # initialize kernels to static init ansatz
    all_Phi, all_G, H0 = initialize_kernels_sampling(Kx, depth, T, nonlin_fn, dnonlin_fn)
    all_A = [jnp.zeros((T,P,T,P)) for l in range(depth)]
    all_B = [jnp.zeros((T,P,T,P)) for l in range(depth)]
    Delta0 = solve_Delta(Kx, y, all_Phi, all_G, eta)
    
    for n in range(num_iter):
        new_Phi = []
        new_G = []
        new_A = []
        new_B = []
        new_V1 = []
        new_V2 = []
        Delta = solve_Delta(Kx, y, all_Phi, all_G, eta)
        # if n % 1 == 0:
        #     plt.plot(jnp.mean(Delta**2 ,axis = 1))
        #     plt.plot(jnp.mean(Delta0**2,axis = 1), '--', color = 'black')
        #     plt.show()
        
        for l in range(depth):
            sys.stdout.write('\r iteration: %d | layer %d / %d' % (n, l+1, depth))
            if l == 0:
                A_minus = jnp.zeros((T,P,T,P))
            else:
                A_minus = all_A[l-1]
            
            if l == depth-1:
                Bl = jnp.zeros((T,P,T,P))
            else:
                Bl = all_B[l]
                
            if l == 0:
                Phi_minus = H0
                S, V = jnp.linalg.eigh(Kx+1e-10)
                S = S * (S > 0.0)
                Z = random.normal(key, (samples, P))
                key, _ = random.split(key)
                chi_1 = V @ jnp.diag( S**(0.5) ) @ Z.T # P x samples
                t = jnp.einsum('ij,k->jki', chi_1, jnp.ones(T))  # samples x T x P

            else:
                Phi_minus = all_Phi[l-1]
                Phi_sqr = Phi_minus.reshape((T*P,T*P))
                S, V = jnp.linalg.eigh(Phi_sqr+1e-6)
                S = S*(S>0.0)
                Z = random.normal(key, (samples, T*P))
                key,_= random.split(key)
                chi_0 = V @ jnp.diag(S**(0.5)) @ Z.T
                chi_0 = chi_0.T # samples x P*T
                t = chi_0.reshape((samples, T, P))
                
            if l == depth - 1:
                G_plus = jnp.ones((T,P,T,P))
                xi_0 = random.normal(key, (samples,))
                r = jnp.einsum('i,jk->ijk', xi_0, jnp.ones((T,P)))
            else:
                G_plus = all_G[l+1]
                G_sqr = G_plus.reshape((T*P,T*P))
                S, V = jnp.linalg.eigh(G_sqr+1e-6)
                S = S*(S>0.0)
                Z = random.normal(key, (samples, P*T))
                key,_ = random.split(key)
                xi_0 = V @ jnp.diag(S**(0.5)) @ Z.T
                xi_0 = xi_0.T
                #xi = xi_0.reshape((samples, T, P)) + jnp.einsum('i,jk->ijk', jnp.ones(samples), jnp.einsum('ijkl,ij->kl', Bl, all_phi_mean[l]))
                r = xi_0.reshape((samples, T, P))
                G_inv = V @ jnp.diag(S**(-1.0)) @ V.T
                G_inv = G_inv.reshape((T,P,T,P))
            
            #print("solving for h, z")
            h, z, chi, xi = solve_self_consistent_four_vars(Phi_minus, G_plus, A_minus, Bl, Delta, t, r, nonlin_fn, dnonlin_fn, eta_gam = eta*gamma)
            #print("h g power")
            #print(jnp.mean(h**2))
            #print(jnp.mean(z**2))
            phi = nonlin_fn(h)
            g = z * dnonlin_fn(h)                        
            
            # solve for jacobians
            #print("solving for jacobian")
            A_new, B_minus_new = solve_jacobians_batched(Phi_minus, G_plus, A_minus, Bl, Delta, h, z, nonlin_fn, dnonlin_fn,ddnonlin_fn,eta_gam=eta*gamma)
            
            # compute <phi(h) phi(h)> and < g g >
            new_Phi += [ 1/samples * jnp.einsum('ijk,ilm->jklm', phi, phi) ]
            new_G += [ 1/samples * jnp.einsum('ijk,ilm->jklm', g, g) ] 
            
            if l < depth-1:
                new_A += [A_new]
                #new_A += [jnp.zeros((T,P,T,P))]
                #print("new A")
                #print(jnp.mean(A_new**2))
            if l > 0:
                new_B += [B_minus_new]
                #new_B += [jnp.zeros((T,P,T,P))]
                #print("new B")
                #print(jnp.mean(B_minus_new**2))
                
            #print("new Phi")
            #print(jnp.mean(new_Phi[-1]**2))
            #print("new G")
            #print(jnp.mean(new_G[-1]**2))                
        
        #print("Vl errors")
        #for l, V1l in enumerate(new_V1):
        #    print(jnp.mean((V1l - new_V2[l])**2))
        
        all_Phi = [alpha*new_Phi[l] + (1-alpha) * all_Phi[l] for l in range(depth)]
        all_G  = [alpha*new_G[l] + (1-alpha) * all_G[l] for l in range(depth)]
        alpha_AB = alpha
        all_A = [alpha_AB * all_A[l] + (1-alpha_AB) * A for l, A in enumerate(new_A)]
        all_B = [alpha_AB * all_B[l] + (1-alpha_AB)* B for l,B in enumerate(new_B)] 

        
    return all_Phi, all_G, all_A, all_B
