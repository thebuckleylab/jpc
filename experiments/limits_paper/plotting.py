import matplotlib.pyplot as plt


def plot_total_energies(energies):
    n_train_iters = len(energies["theory"])
    train_iters = [b+1 for b in range(n_train_iters)]
    
    _, ax = plt.subplots(figsize=(6, 3))
    
    for energy_type, energy in energies.items():
        is_theory = energy_type == "theory"
        line_style = "--" if is_theory else "-"
        color = "black" if is_theory else "#00CC96"  #"rgb(27, 158, 119)"
        
        if color.startswith("rgb"):
            rgb = tuple(int(x)/255 for x in color[4:-1].split(","))
        else:
            rgb = color
    
        ax.plot(
            train_iters, 
            energy, 
            label=energy_type, 
            linewidth=4 if is_theory else 3,
            linestyle=line_style,
            color=rgb
        )
    
    ax.legend(fontsize=16)
    ax.set_xlabel("Training Iteration", fontsize=18, labelpad=10)
    ax.set_ylabel("Energy", fontsize=18, labelpad=10)
    ax.tick_params(axis='both', labelsize=14)
    plt.grid(True)
    plt.show()
