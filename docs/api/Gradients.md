# Gradients

!!! info
    There are two similar functions to compute the gradient of the energy with
    respect to the activities: [`jpc.neg_activity_grad()`](http://127.0.0.1:8000/api/Gradients/#jpc.neg_activity_grad) 
    and [`jpc.compute_activity_grad()`](http://127.0.0.1:8000/api/Gradients/#jpc.compute_activity_grad). 
    The first is used by [`jpc.solve_inference()`](http://127.0.0.1:8000/api/Continuous%20Inference/#jpc.solve_inference) 
    as gradient flow, while the second is for compatibility with discrete 
    [optax](https://github.com/google-deepmind/optax) optimisers such as 
    gradient descent.

::: jpc.neg_activity_grad

---

::: jpc.compute_activity_grad

---

::: jpc.compute_pc_param_grads

---

::: jpc.compute_hpc_param_grads