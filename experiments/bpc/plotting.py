import io
import matplotlib.pyplot as plt
from PIL import Image
import jax.numpy as jnp


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def get_fashion_mnist_label(label):
    return {
        "0": "T-shirt/top",
        "1": "Trouser",
        "2": "Pullover",
        "3": "Dress",
        "4": "Coat",
        "5": "Sandal",
        "6": "Shirt",
        "7": "Sneaker",
        "8": "Bag",
        "9": "Ankle boot"
    }[label]


def plot_imgs(imgs, dataset, labels, train_step, n_imgs=10):
    fig, axes = plt.subplots(1, n_imgs, figsize=(20, 2))
    fig.suptitle(f"Training step {train_step}", fontsize=16, y=1.02)
    
    for i, ax in enumerate(axes[:n_imgs]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(imgs[i].reshape(28, 28), cmap=plt.cm.binary_r)
        label = str(jnp.argmax(labels, axis=1)[i])
        ax.set_xlabel(
            label if dataset == "MNIST" else get_fashion_mnist_label(label), 
            fontsize=16
        )
    
    return fig
