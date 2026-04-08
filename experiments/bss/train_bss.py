"""
Train a one-layer regularised PDM on blind source separation (BSS).

Generates uncorrelated uniform sources, mixes them with a random matrix,
adds optional noise, then trains PDM with x = mixtures, y = sources. Evaluates
with SINR (and optionally per-source SNR) after permutation/sign correction.
"""

import argparse
import os
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax

import jpc


# -----------------------------------------------------------------------------
# BSS data generation
# -----------------------------------------------------------------------------

def generate_uncorrelated_uniform_sources(n_sources, n_samples, low=-1.0, high=1.0):
    return np.random.uniform(
        low=low, high=high, size=(n_sources, n_samples)
    ).astype(np.float32)


def add_white_gaussian_noise(X_noiseless, snr_db):
    """
    Add white Gaussian noise to achieve target signal to noise ratio in dB.
    """
    signal_power = np.mean(X_noiseless ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power)
    noise = np.random.randn(*X_noiseless.shape).astype(np.float32) * sigma
    return X_noiseless + noise


def display_matrix(A):
    print(np.round(A, 4))


# -----------------------------------------------------------------------------
# BSS evaluation
# -----------------------------------------------------------------------------

def compute_snr_np(S_original, S_noisy):
    """SNR per channel in dB. S_original, S_noisy: (n_sources, n_samples)."""
    N_hat = S_original - S_noisy
    N_P = (N_hat ** 2).sum(axis=1)
    S_P = (S_original ** 2).sum(axis=1)
    return 10 * np.log10(S_P / (N_P + 1e-12))


def outer_prod_broadcasting(A, B):
    """Pairwise outer product between columns of two matrices for correlation."""
    return A[..., None] * B[:, None]


def find_permutation_between_source_and_estimation(S, Y):
    """Best 1-to-1 mapping from estimated Y to ground-truth S (by correlation)."""
    correlation_numerator = np.abs(outer_prod_broadcasting(Y.T, S.T).sum(axis=0))
    normalization = np.linalg.norm(S, axis=1) * np.linalg.norm(Y, axis=1)
    normalization = np.maximum(normalization, 1e-12)
    perm = np.argmax(correlation_numerator / normalization, axis=0)
    return perm


def signed_and_permutation_corrected_sources(S, Y):
    """Reorder Y and flip signs to match S."""
    perm = find_permutation_between_source_and_estimation(S, Y)
    matched_Y = Y[perm, :]
    signs = np.sign((matched_Y * S).sum(axis=1))
    signs[signs == 0] = 1
    return signs[:, np.newaxis] * matched_Y


def compute_sinr_np(S, Y):
    """
    SINR in dB with internal z-score normalization and permutation/sign correction.
    S, Y: (n_sources, n_samples).
    """
    N_sources = S.shape[0]
    S_normalized = np.zeros_like(S)
    Y_normalized = np.zeros_like(Y)
    for i in range(N_sources):
        s_i = S[i, :]
        S_normalized[i, :] = (s_i - np.mean(s_i)) / (np.std(s_i) + 1e-9)
        y_i = Y[i, :]
        Y_normalized[i, :] = (y_i - np.mean(y_i)) / (np.std(y_i) + 1e-9)
    corr = np.dot(Y_normalized, S_normalized.T)
    perm = np.array([np.argmax(np.abs(corr[:, i])) for i in range(N_sources)])
    signs = np.sign(corr[perm, np.arange(N_sources)])
    Y_normalized_corrected = signs[:, np.newaxis] * Y_normalized[perm, :]
    E = Y_normalized_corrected - S_normalized
    MSE = np.linalg.norm(E) ** 2
    SigPow = np.linalg.norm(S_normalized) ** 2
    return 10 * np.log10(SigPow / (MSE + 1e-12))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def main(
    n_samples,
    n_sources,
    n_mixtures,
    snr_db,
    activity_lr,
    param_lr,
    n_infer_iters,
    use_lateral,
    lambda_param,
    test_every,
    save_dir,
    seed,
    n_val
):
    key = jr.PRNGKey(seed)
    k_w, k_perm = jr.split(key, 2)

    # Data: train (n_samples) + validation (n_val). Sources (S, N_total), mixing M, mixtures X.
    n_total = n_samples + n_val
    S_all = generate_uncorrelated_uniform_sources(n_sources, n_total)
    M = np.random.randn(n_mixtures, n_sources).astype(np.float32)
    X_noiseless_all = M @ S_all
    X_all = add_white_gaussian_noise(X_noiseless_all, snr_db)

    S = S_all[:, :n_samples]
    S_val = S_all[:, n_samples:]
    X = X_all[:, :n_samples]
    X_val = X_all[:, n_samples:]

    SNRinp = 10 * np.log10(
        np.sum(np.mean(X_noiseless_all ** 2, axis=1))
        / (np.sum(np.mean((X_noiseless_all - X_all) ** 2, axis=1)) + 1e-12)
    )
    print("Correlation matrix of sources (train):")
    display_matrix(np.corrcoef(S))
    print(f"Input signal to noise ratio (dB): {SNRinp:.2f}")
    print(f"BSS with lateral={use_lateral}  train={n_samples}  val={n_val}")

    # BSS: x = mixtures (N, n_mixtures), z = latent sources (N, n_sources), W = demixing (n_sources, n_mixtures)
    W = jr.normal(k_w, (n_sources, n_mixtures)).astype(jnp.float32) * 0.01
    if use_lateral:
        L = -jnp.ones((n_sources, n_sources), dtype=jnp.float32) * 0.1
        L = L.at[jnp.diag_indices(n_sources)].set(0.0)
    else:
        L = None

    # Train data: (n_samples, n_mixtures)
    X_jax = jnp.array(X.T)
    X_val_jax = jnp.array(X_val.T)
    N = n_samples

    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)
    z_placeholder = jnp.zeros((1, n_sources), dtype=jnp.float32)
    activity_opt_state = activity_optim.init(z_placeholder)
    param_opt_state = param_optim.init(W)

    train_losses = []
    sinrs = []

    perm = jr.permutation(k_perm, N)
    for step in range(N):
        idx = perm[step]
        x_batch = X_jax[idx : idx + 1]

        # Inference
        z = x_batch @ W.T
        activity_opt_state = activity_optim.init(z)
        for _ in range(n_infer_iters):
            act_kwargs = dict(
                W=W,
                z=z,
                x=x_batch,
                optim=activity_optim,
                opt_state=activity_opt_state,
            )
            if use_lateral and L is not None:
                act_kwargs["L"] = L
            
            result = jpc.update_bss_activities(**act_kwargs)
            z = result["z"]
            activity_opt_state = result["opt_state"]

        train_energy = float(result["energy"])
        train_losses.append(train_energy)

        # Update W, and L via moving average
        param_kwargs = dict(
            W=W,
            z=z,
            x=x_batch,
            optim=param_optim,
            opt_state=param_opt_state,
        )
        if use_lateral and L is not None and lambda_param is not None:
            param_kwargs["L"] = L
            param_kwargs["lambda_param"] = lambda_param
        
        param_result = jpc.update_bss_params(**param_kwargs)
        W = param_result["W"]
        param_opt_state = param_result["opt_state"]
        if use_lateral and "L" in param_result:
            L = param_result["L"]

        if (step + 1) % test_every == 0 or step == 0:
            Y_est_list = []
            for i in range(n_val):
                x_one = X_val_jax[i : i + 1]
                z_eval = x_one @ W.T
                opt_state_eval = activity_optim.init(z_eval)
                for _ in range(n_infer_iters):
                    eval_kwargs = dict(
                        W=W,
                        z=z_eval,
                        x=x_one,
                        optim=activity_optim,
                        opt_state=opt_state_eval,
                    )
                    if use_lateral and L is not None:
                        eval_kwargs["L"] = L
                    result_eval = jpc.update_bss_activities(**eval_kwargs)
                    z_eval = result_eval["z"]
                    opt_state_eval = result_eval["opt_state"]
                Y_est_list.append(np.array(z_eval.T))
            
            Y_est = np.concatenate(Y_est_list, axis=1)
            sinr = compute_sinr_np(S_val, Y_est)
            sinrs.append(sinr)
            # Smoothed training loss (mean over last test_every steps) for less noisy logging
            recent = train_losses[-test_every:] if len(train_losses) >= test_every else train_losses
            train_mean = float(np.mean(recent))
            print(
                f"Step {step + 1}/{N}  train_energy = {train_energy:.6f}  train_mean = {train_mean:.6f}  val SINR (dB) = {sinr:.2f}"
            )

    os.makedirs(save_dir, exist_ok=True)
    train_losses_arr = np.array(train_losses)
    np.save(os.path.join(save_dir, "train_losses.npy"), train_losses_arr)
    # Smoothed loss (rolling mean) for plotting; same length as train_losses
    window = min(50, max(1, len(train_losses_arr) // 20))
    train_losses_smooth = np.convolve(train_losses_arr, np.ones(window) / window, mode="same")
    np.save(os.path.join(save_dir, "train_losses_smooth.npy"), train_losses_smooth)
    np.save(os.path.join(save_dir, "sinrs.npy"), np.array(sinrs))
    np.save(os.path.join(save_dir, "W.npy"), np.array(W))
    if L is not None:
        np.save(os.path.join(save_dir, "L.npy"), np.array(L))

    # Final evaluation on validation set
    Y_final_list = []
    for i in range(n_val):
        x_one = X_val_jax[i : i + 1]
        z_eval = x_one @ W.T
        opt_state_eval = activity_optim.init(z_eval)
        for _ in range(n_infer_iters):
            eval_kwargs = dict(
                W=W,
                z=z_eval,
                x=x_one,
                optim=activity_optim,
                opt_state=opt_state_eval,
            )
            if use_lateral and L is not None:
                eval_kwargs["L"] = L
            result_eval = jpc.update_bss_activities(**eval_kwargs)
            z_eval = result_eval["z"]
            opt_state_eval = result_eval["opt_state"]
        Y_final_list.append(np.array(z_eval.T))

    Y_final = np.concatenate(Y_final_list, axis=1)
    Y_corrected = signed_and_permutation_corrected_sources(S_val, Y_final)
    snr_per_source = compute_snr_np(S_val, Y_corrected)
    final_sinr = compute_sinr_np(S_val, Y_final)
    print(f"\nFinal val SINR (dB): {final_sinr:.2f}")
    print("Per-source SNR (dB) [val]:", snr_per_source)
    np.save(os.path.join(save_dir, "snr_per_source.npy"), snr_per_source)
    np.save(os.path.join(save_dir, "ground_truth_sources.npy"), S_val)
    np.save(os.path.join(save_dir, "reconstructed_sources.npy"), Y_corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="bss_results")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--n_mixtures", type=int, default=50, help="Default: n_sources + 2")
    parser.add_argument("--snr_db", type=float, default=0)
    parser.add_argument("--activity_lr", type=float, default=1e-3)
    parser.add_argument("--param_lr", type=float, default=1e-4)
    parser.add_argument("--n_infer_iters", type=int, default=100)
    parser.add_argument("--use_lateral", action="store_true", default=True)
    parser.add_argument("--lambda_param", type=float, default=0.9)
    parser.add_argument("--test_every", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_mixtures = args.n_mixtures if args.n_mixtures is not None else args.n_sources + 2
    save_dir = os.path.join(
        args.results_dir,
        f"n_samples_{args.n_samples}",
        f"n_sources_{args.n_sources}",
        f"n_mixtures_{n_mixtures}",
        f"snr_{args.snr_db}",
        f"{args.activity_lr}_activity_lr",
        f"{args.param_lr}_param_lr",
        f"{args.n_infer_iters}_infer_iters",
        f"lateral_{args.use_lateral}",
        f"n_val_{args.n_val}",
        str(args.seed),
    )
    main(
        n_samples=args.n_samples,
        n_sources=args.n_sources,
        n_mixtures=n_mixtures,
        snr_db=args.snr_db,
        activity_lr=args.activity_lr,
        param_lr=args.param_lr,
        n_infer_iters=args.n_infer_iters,
        use_lateral=args.use_lateral,
        lambda_param=args.lambda_param,
        test_every=args.test_every,
        save_dir=save_dir,
        seed=args.seed,
        n_val=args.n_val,
    )
