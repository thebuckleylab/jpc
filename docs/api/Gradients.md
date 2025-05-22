# Gradients

!!! note
    There are two similar functions to compute the activity gradient: 
    `jpc.neg_activity_grad()` and `jpc.compute_activity_grad()`. The first is 
    used by `jpc.solve_inference()` as gradient flow, while the second is for 
    compatibility with discrete optax optimisers such as gradient descent.

::: jpc.neg_activity_grad

---

::: jpc.compute_activity_grad

---

::: jpc.compute_pc_param_grads

---

::: jpc.compute_hpc_param_grads