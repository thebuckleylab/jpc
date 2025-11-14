# Gradients

!!! info
    There are two similar functions to compute the gradient of the energy with
    respect to the activities of a standard PC energy: [`jpc.neg_pc_activity_grad()`](https://thebuckleylab.github.io/jpc/api/Gradients/#jpc.neg_pc_activity_grad) 
    and [`jpc.compute_pc_activity_grad()`](https://thebuckleylab.github.io/jpc/api/Gradients/#jpc.compute_pc_activity_grad). 
    The first is used by [`jpc.solve_inference()`](https://thebuckleylab.github.io/jpc/api/Continuous-time%20Inference/#jpc.solve_inference) 
    as gradient flow, while the second is for compatibility with discrete 
    [optax](https://github.com/google-deepmind/optax) optimisers such as 
    gradient descent.

::: jpc.neg_pc_activity_grad

---

::: jpc.compute_pc_activity_grad

---

::: jpc.compute_bpc_activity_grad

---

::: jpc.compute_pc_param_grads

---

::: jpc.compute_hpc_param_grads

---

::: jpc.compute_bpc_param_grads