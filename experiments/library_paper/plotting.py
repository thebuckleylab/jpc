import jax.numpy as jnp
import plotly.graph_objs as go
import plotly.colors as pc
from utils import compute_metric_stats


def plot_loss(loss, yaxis_title, xaxis_title, save_path):
    n_train_iters = len(loss)
    train_iters = [t+1 for t in range(n_train_iters)]

    loss_color = "#EF553B"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=loss,
            mode="lines",
            line=dict(width=2, color=loss_color),
            showlegend=False
        )
    )
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title=xaxis_title,
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_accuracy(accuracy, yaxis_title, xaxis_title, save_path):
    n_train_iters = len(accuracy)
    train_iters = [t+1 for t in range(n_train_iters)]

    accuracy_color = "#636EFA"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=accuracy,
            mode="lines",
            line=dict(width=2, color=accuracy_color),
            showlegend=False
        )
    )
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title=xaxis_title,
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_energies(energies, save_path):
    energies = jnp.flip(
        jnp.array(energies).T,
        axis=0
    )
    n_layers = energies.shape[0]
    n_train_iters = energies.shape[1]
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    colors = pc.sample_colorscale("Oranges", n_layers)
    for i, (energy, color) in enumerate(zip(energies, colors)):
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=energy,
                name=f"$l_{{{i+1}}}$",
                mode="lines",
                line=dict(width=2, color=color)
            )
        )

    fig.update_layout(
        height=400,
        width=650,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]],
        ),
        yaxis=dict(
            title="Energy",
            nticks=3
        ),
        font=dict(size=16),
    )
    fig.write_image(save_path)


def plot_norms(norms, norm_type, save_path):
    norms = jnp.array(norms).T
    n_layers = norms.shape[0]
    n_train_iters = norms.shape[1]
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    colorscale = "Reds" if norm_type == "activity" else "Greens"
    colors = pc.sample_colorscale(colorscale, n_layers+2)[2:]
    for l, (norm, color) in enumerate(zip(norms, colors)):
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=norm,
                name=f"$l_{{{l+1}}}$",
                mode="lines",
                line=dict(width=2, color=color)
            )
        )
    if norm_type == "param":
        yaxis_title = "$\Large{||\\theta_l||_2}$"
    elif norm_type == "param_grad":
        yaxis_title = "$\Large{||\partial \\theta_l||_2}$"
    elif norm_type == "activity":
        yaxis_title = "$\Large{||\mathbf{z}_l||_2}$"

    fig.update_layout(
        height=500,
        width=700,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]],
        ),
        yaxis=dict(title=yaxis_title),
        font=dict(size=16),
        margin=dict(r=120)
    )
    fig.write_image(save_path)


def plot_activity_norms(norms, save_path):
    num_norms, theory_norms = norms
    n_layers = num_norms.shape[1]
    n_train_iters = num_norms.shape[0]
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            name="theory"
        )
    )
    colors = pc.sample_colorscale("Reds", n_layers+2)[2:]
    for i, color in enumerate(colors):
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=num_norms[:, i],
                name=f"$l_{{{i+1}}}$",
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.8
            )
        )
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=theory_norms[:, i],
                mode="lines",
                line=dict(width=4, color=color, dash="dash"),
                showlegend=False
            )
        )
    fig.update_layout(
        height=500,
        width=700,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]],
        ),
        yaxis=dict(title="$\Large{||\mathbf{z}_l||_2}$"),
        font=dict(size=16),
        margin=dict(r=120)
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
    n_train_iters = len(loss)
    train_iters = [t+1 for t in range(n_train_iters)]

    loss_color, accuracy_color = "#EF553B", "#636EFA"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=loss,
            mode="lines",
            line=dict(width=2, color=loss_color),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=accuracy,
            mode="lines",
            line=dict(width=2, color=accuracy_color),
            showlegend=False,
            yaxis="y2"
        )
    )
    xticks = [1, int(train_iters[-1]/2), train_iters[-1]] if (
        test_every == 1
    ) else [t for t in train_iters if t == 1 or t % test_every == 0]
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title=xaxis_title,
            tickvals=xticks,
            ticktext=xticks
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


def plot_runtimes(runtimes, save_path):
    # skip first runtime when jit compilation happens
    runtimes = runtimes[1:]
    n_train_iters = len(runtimes)
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_iters,
            y=runtimes,
            mode="lines",
            line=dict(width=2, color="#FFA15A"),
            showlegend=False
        )
    )
    fig.update_layout(
        height=300,
        width=400,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(
            title="Runtime (ms)"
        ),
        font=dict(size=16)
    )
    fig.write_image(save_path)


def plot_runtime_stats(solvers_runtime, save_path, test_every=1):
    fig = go.Figure()
    colors = pc.sample_colorscale("Blues", len(solvers_runtime) + 2)[::-1]
    max_train_iter = 0
    for i, solver_id in enumerate(solvers_runtime.keys()):

        means, stds = compute_metric_stats(solvers_runtime[solver_id])
        means, stds = means[1:], stds[1:]  # skip first runtime when jit compilation happens
        n_train_iters = len(means)
        train_iters = [t + 1 for t in range(n_train_iters)]
        if n_train_iters > max_train_iter:
            max_train_iter = n_train_iters

        y_upper, y_lower = means + stds, means - stds
        fig.add_traces(
            go.Scatter(
                x=list(train_iters) + list(train_iters[::-1]),
                y=list(y_upper) + list(y_lower[::-1]),
                fill="toself",
                fillcolor=colors[i],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.3
            )
        )
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=means,
                mode="lines",
                name=solver_id,
                line=dict(width=2, color=colors[i]),
            )
        )

    xtickvals = [t + 1 for t in range(max_train_iter) if t == 0 or (t + 1) % test_every == 0]
    xticktext = [str(t * test_every) for t in xtickvals]

    fig.update_layout(
        height=400,
        width=600,
        xaxis=dict(
            title="Training iteration",
            tickvals=xtickvals,
            ticktext=xticktext,
            nticks=3
        ),
        yaxis=dict(title="Runtime (ms)"),
        font=dict(size=16),
        margin=dict(r=120)
    )
    fig.write_image(save_path)


def plot_total_energies(energies, save_path):
    n_train_iters = len(energies[0])
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    for i, energy in enumerate(energies):
        is_theory = i == 1
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=energy,
                name="theory" if is_theory else "experiment",
                mode="lines",
                line=dict(
                    width=3,
                    dash="dash" if is_theory else "solid",
                    color="rgb(27, 158, 119)" if is_theory else "#00CC96"
                ),
                legendrank=1 if is_theory else 2
            )
        )

    fig.update_layout(
        height=300,
        width=450,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]],
        ),
        yaxis=dict(
            title="Energy",
            nticks=3
        ),
        font=dict(size=16),
    )
    fig.write_image(save_path)


def plot_layer_energies(energies, save_path):
    n_layers = energies[0].shape[1]
    n_train_iters = energies[0].shape[0]
    train_iters = [t+1 for t in range(n_train_iters)]

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=2, color="black", dash="dash"),
            name="theory"
        )
    )
    colors = pc.sample_colorscale("Greens", n_layers+3)[3:]
    for i, color in enumerate(colors):
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=energies[0][:, i],
                name=f"$l_{{{i+1}}}$",
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.8
            )
        )
        fig.add_traces(
            go.Scatter(
                x=train_iters,
                y=energies[1][:, i],
                mode="lines",
                line=dict(width=4, color=color, dash="dash"),
                showlegend=False
            )
        )

    fig.update_layout(
        height=400,
        width=600,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]],
        ),
        yaxis=dict(
            title="Energy",
            nticks=3
        ),
        font=dict(size=16),
        margin=dict(r=120)
    )
    fig.write_image(save_path)
