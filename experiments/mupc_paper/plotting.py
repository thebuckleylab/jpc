import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.colors as pc
from utils import compute_metric_stats


def plot_activity_hessian(coeff_matrix, save_path, title=None):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        X=coeff_matrix,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )
    cbar = fig.colorbar(heatmap, ax=ax, location="right", ticks=[-1, 0, 1])
    cbar.ax.tick_params(labelsize=25)

    if len(coeff_matrix) > 10:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    else:
        ticks = np.arange(len(coeff_matrix), dtype=int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks + 1)
        ax.set_yticklabels(ticks + 1)

    if title is not None:
        plt.title(title, fontsize=20)

    fig.savefig(save_path)
    plt.close(fig)


def plot_loss(loss, yaxis_title, xaxis_title, save_path, mode="lines+markers"):
    n_train_iters = len(loss)
    train_iters = [t for t in range(n_train_iters)]

    loss_color = "#EF553B"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=loss,
            mode=mode,
            line=dict(width=2, color=loss_color),
            showlegend=False
        )
    )
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title=xaxis_title,
            tickvals=[0, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[0, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_n_infer_iters(n_infer_iters, save_path):
    n_train_iters = len(n_infer_iters)
    train_iters = [t for t in range(n_train_iters)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=n_infer_iters,
            mode="lines",
            line=dict(width=2, color="#FFA15A"),
            showlegend=False
        )
    )
    fig.update_layout(
        height=350,
        width=525,
        xaxis=dict(
            title="Training iteration",
            tickvals=[0, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[0, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(title="# infer iterations"),
        font=dict(size=16),
        margin=dict(b=90)
    )
    fig.write_image(save_path)


def plot_loss_and_accuracy(
        loss,
        accuracy,
        mode,
        xaxis_title,
        save_path,
        test_every=1
):
    n_iters = len(loss)
    iters = [t for t in range(n_iters)]

    loss_color, accuracy_color = "#EF553B", "#636EFA"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=loss,
            mode="lines+markers",
            line=dict(width=2, color=loss_color),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=accuracy,
            mode="lines+markers",
            line=dict(width=2, color=accuracy_color),
            showlegend=False,
            yaxis="y2"
        )
    )
    xtickvals = [0, int(iters[-1]/2), iters[-1]]
    xticktext = xtickvals if (
            test_every == 1
    ) else [(t+1)*test_every for t in xtickvals]
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title=xaxis_title,
            tickvals=xtickvals,
            ticktext=xticktext
        ),
        yaxis=dict(
            title=f"{mode.capitalize()} loss",
            titlefont=dict(
                color=loss_color
            ),
            tickfont=dict(
                color=loss_color
            )
        ),
        yaxis2=dict(
            title=f"{mode.capitalize()} accuracy (%)",
            side="right",
            overlaying="y",
            titlefont=dict(
                color=accuracy_color
            ),
            tickfont=dict(
                color=accuracy_color,
            )
        ),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_norms(
        norms,
        norm_type,
        save_path,
        theory_norms=None,
        log=False,
        showticklabels=True
):
    norms = np.array(norms).T
    n_layers = norms.shape[0]
    n_train_iters = norms.shape[1]
    train_iters = [t for t in range(n_train_iters)]

    fig = go.Figure()
    if theory_norms is not None:
        fig.add_traces(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(width=3, color="black", dash="dash"),
                name="theory"
            )
        )

    layer_idxs = [1, "1/4L", "1/2L", "3/4L", "L"]
    colorscale = "Reds" if norm_type == "activity" else "Greens"
    colors = pc.sample_colorscale(colorscale, n_layers + 2)[2:]
    for i, norm in enumerate(norms):
        layer_idx = layer_idxs[i]
        if n_train_iters == 1:
            fig.add_hline(
                y=norm[0],
                name=f"$\ell = {{{layer_idx}}}$",
                line=dict(
                    width=2,
                    color=colors[i]
                )
            )
        else:
            fig.add_traces(
                go.Scatter(
                    x=train_iters,
                    y=norm,
                    name=f"$\ell = {{{layer_idx}}}$",
                    mode="lines",
                    line=dict(
                        width=2,
                        color=colors[i]
                    )
                )
            )
        if theory_norms is not None and norm_type == "activity":
            fig.add_hline(
                y=theory_norms[i],
                line=dict(
                    color=colors[i],
                    width=4,
                    dash="dash"
                )
            )

    if norm_type == "param_l2":
        yaxis_title = "$\Large{||W_\ell||_F}$"
    elif norm_type == "param_spectral":
        yaxis_title = "$\Large{||W_\ell||_2}$"
    elif norm_type == "activity":
        yaxis_title = "$\Large{||\mathbf{z}_\ell||_2}$"

    lmargin = 100 if (log and norm_type == "activity") else 80
    fig.update_layout(
        height=350,
        width=525,
        xaxis=dict(
            title="Inference iteration" if (
                    norm_type == "activity"
            ) else "Training iteration",
            tickvals=[0, int(train_iters[-1] / 2), train_iters[-1]],
            ticktext=[0, int(train_iters[-1] / 2), train_iters[-1]],
            showticklabels=showticklabels
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=18),
        margin=dict(r=140, l=lmargin, b=90)
    )
    if log:
        fig.update_layout(
            yaxis=dict(
                type="log",
                exponentformat="power",
                dtick=1
            )
        )
    fig.write_image(save_path)


def plot_energies(energies, test_every, save_path, theory_energies=None, log=False):
    n_layers = energies.shape[0]
    n_iters = energies.shape[1]
    iters = [t for t in range(n_iters)]

    fig = go.Figure()
    if theory_energies is not None:
        fig.add_traces(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(width=3, color="black", dash="dash"),
                name="theory"
            )
        )

    layer_idxs = [1, "1/4L", "1/2L", "3/4L", "L"]
    colors = pc.sample_colorscale("Purples", n_layers + 3)[3:]
    for i, color in enumerate(colors):
        layer_idx = layer_idxs[i]
        fig.add_traces(
            go.Scatter(
                x=iters,
                y=energies[i],
                name=f"$\ell = {{{layer_idx}}}$",
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.8
            )
        )
        if theory_energies is not None:
            fig.add_traces(
                go.Scatter(
                    x=iters,
                    y=theory_energies[i],
                    mode="lines",
                    line=dict(width=3, color=color, dash="dash"),
                    showlegend=False
                )
            )

    xtickvals = [0, int(iters[-1] / 2), iters[-1]]
    xticktext = [t * test_every for t in xtickvals]
    fig.update_layout(
        height=350,
        width=525,
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext,
        ),
        yaxis=dict(title="Energy"),
        font=dict(size=18),
        margin=dict(r=140, b=90)
    )
    if log:
        fig.update_layout(
            yaxis=dict(
                title="Energy (log)",
                type="log",
                exponentformat="power",
                dtick=0
            )
        )
    fig.write_image(save_path)


def plot_hessian_eigenvalues_during_training(eigenvals, test_every, save_path):
    colors = pc.sample_colorscale("Oranges", len(eigenvals) + 2)[2:]
    n_bins = 50
    fig = go.Figure()
    for i, eigens in enumerate(eigenvals):
        t = i * test_every
        fig.add_trace(
            go.Histogram(
                x=eigens,
                histnorm="probability",
                nbinsx=n_bins,
                name=f"$t = {t}$",
                marker=dict(color=colors[i])
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=350,
        width=525,
        xaxis=dict(title="$\LARGE{\lambda(\mathrm{H}_{\mathbf{z}})}$"),
        yaxis=dict(
            title=f"Density (log)",
            type="log",
            exponentformat="power",
            dtick=1
        ),
        font=dict(size=18),
        margin=dict(b=90)
    )
    fig.update_traces(opacity=0.75)
    fig.write_image(save_path)


def plot_max_min_eigenvals(max_min_eigenvals, test_every, save_path):
    n_iters = max_min_eigenvals.shape[-1]
    iters = [t for t in range(n_iters)]

    labels = ["$\lambda_{max}$", "$\lambda_{min}$"]
    colors = ["#EF553B", "#636EFA"]
    fig = go.Figure()
    for i in range(2):
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=max_min_eigenvals[i],
                mode="lines+markers",
                line=dict(width=2, color=colors[i]),
                name=labels[i]
            )
        )

    xtickvals = [0, int(iters[-1] / 2), iters[-1]]
    xticktext = [t * test_every for t in xtickvals]
    fig.update_layout(
        height=350,
        width=525,
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext
        ),
        yaxis=dict(
            title="$\LARGE{\lambda(\mathrm{H}_{\mathbf{z}})}$",
            type="log",
            exponentformat="power",
            dtick=1
        ),
        font=dict(size=18),
        margin=dict(l=100, r=140, b=90)
    )
    fig.write_image(save_path)


def plot_max_min_eigenvals_2_axes(max_min_eigenvals, test_every, save_path):
    n_iters = max_min_eigenvals.shape[-1]
    iters = [t for t in range(n_iters)]

    max_color, min_color = "#EF553B", "#636EFA"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=max_min_eigenvals[0],
            mode="lines+markers",
            line=dict(width=2, color=max_color),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=max_min_eigenvals[1],
            mode="lines+markers",
            line=dict(width=2, color=min_color),
            showlegend=False,
            yaxis="y2"
        )
    )

    xtickvals = [0, int(iters[-1] / 2), iters[-1]]
    xticktext = [t * test_every for t in xtickvals]
    fig.update_layout(
        height=350,
        width=525,
        margin=dict(r=140, b=90),
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext
        ),
        yaxis=dict(
            title="$\Large{\lambda_{max}}$",
            titlefont=dict(
                color=max_color
            ),
            tickfont=dict(
                color=max_color
            )
        ),
        yaxis2=dict(
            title="$\LARGE{\lambda_{min}}$",
            side="right",
            overlaying="y",
            titlefont=dict(
                color=min_color
            ),
            tickfont=dict(
                color=min_color,
            )
        ),
        font=dict(size=18)
    )
    fig.write_image(save_path)


def plot_cond_num_stats(cond_nums, test_every, save_path):
    means, stds = compute_metric_stats(cond_nums)
    y_upper, y_lower = means + stds, means - stds

    n_iters = len(means)
    iters = [t for t in range(n_iters)]

    color = "#FFA15A"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(iters) + list(iters[::-1]),
            y=list(y_upper) + list(y_lower[::-1]),
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.3
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=means,
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False
        )
    )

    xtickvals = [0, int(iters[-1] / 2), iters[-1]]
    xticktext = [t * test_every for t in xtickvals]
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext
        ),
        yaxis=dict(title="$\Large{\kappa(\mathrm{H}_{\mathbf{z}})}$"),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_metric_stats(metric, metric_id, test_every, save_path):
    means, stds = compute_metric_stats(metric)
    y_upper, y_lower = means + stds, means - stds

    n_iters = len(means)
    iters = [t for t in range(n_iters)]

    color = "#636EFA" if metric_id == "test_acc" else "#FFA15A"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(iters) + list(iters[::-1]),
            y=list(y_upper) + list(y_lower[::-1]),
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.3
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=means,
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False
        )
    )

    xtickvals = [0, int(iters[-1] / 2), iters[-1]]
    xticktext = [(t+1) * test_every for t in xtickvals]
    yaxis_title = "Test accuracy (%)" if (
        metric_id == "test_acc"
    ) else "$\Large{\kappa(\mathrm{H}_{\mathbf{z}})}$"
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_activities(activities, save_path, theory_activities=None, log=False):
    n_layers = activities.shape[1]
    n_train_iters = activities.shape[0]
    train_iters = [t for t in range(n_train_iters)]
    layer_idxs = [1, "1/4L", "1/2L", "3/4L", "L"]
    
    fig = go.Figure()
    if theory_activities is not None:
        fig.add_traces(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(width=3, color="black", dash="dash"),
                name="theory"
            )
        )

    colorscale = "Reds"
    colors = pc.sample_colorscale(colorscale, n_layers + 2)[2:]
    for i, color in enumerate(colors):
        layer_idx = layer_idxs[i]

        if n_train_iters == 1:
            fig.add_hline(
                y=activities[0, i],
                name=f"$\ell = {{{layer_idx}}}$",
                line=dict(
                    color=color,
                    width=2
                ),
                showlegend=True
            )
        else:
            fig.add_traces(
                go.Scatter(
                    x=train_iters,
                    y=activities[:, i],
                    name=f"$\ell = {{{layer_idx}}}$",
                    mode="lines",
                    line=dict(
                        width=2,
                        color=color
                    )
                )
            )
        
        if theory_activities is not None:
            fig.add_hline(
                y=theory_activities[i],
                line=dict(
                    color=color,
                    width=4,
                    dash="dash"
                )
            )

    fig.update_layout(
        height=350,
        width=525,
        xaxis=dict(
            title="Inference iteration",
            showticklabels=False
        ),
        yaxis=dict(title="$\Large{z_\ell}$"),
        font=dict(size=18),
        margin=dict(r=140, l=100, b=90)
    )
    if log:
        fig.update_layout(
            yaxis=dict(
                type="log",
                exponentformat="power",
                dtick=1
            )
        )
    if n_train_iters > 1:
        fig.update_layout(
            xaxis=dict(
                showticklabels=True,
                tickvals=[0, int(train_iters[-1] / 2), train_iters[-1]],
                ticktext=[0, int(train_iters[-1] / 2), train_iters[-1]]
            )
        )
    
    fig.write_image(save_path)


def plot_2D_data(x, y, y_hat, save_path):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[:, 0],
            y=y[:, 0],
            mode="markers",
            marker=dict(size=8, color="#636EFA"),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[:, 0],
            y=y_hat[:, 0],
            mode="markers",
            marker=dict(size=8, color="#EF553B"),
            name="$\hat{y}$"
        )
    )
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(title="$\Large{x}$"),
        yaxis=dict(title="$\Large{y}$"),
        font=dict(size=16)
    )
    fig.write_image(save_path)
