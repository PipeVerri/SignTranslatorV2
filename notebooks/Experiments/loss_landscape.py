import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from contextlib import nullcontext
import plotly.graph_objects as go

@torch.no_grad()
def compute_and_plot_loss_surface(
    model,
    test_loader,
    criterion,
    direction1,
    direction2,
    base_params=None,
    grid_size=100,
    range_limit=1e5,
    device=None,
    samples_per_eval=20,
    use_amp=True,
    compile_model=True,
    show_plot=True
):
    """
    Compute and render a 3D loss surface for a model along two parameter directions.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    test_loader : DataLoader
        DataLoader providing test data.
    criterion : callable
        Loss function.
    direction1, direction2 : torch.Tensor
        Direction vectors (same shape as model parameters vector).
    base_params : torch.Tensor, optional
        Base parameter vector. If None, taken from the model.
    grid_size : int, default=100
        Number of grid points per dimension.
    range_limit : float, default=1e5
        Range for alpha and beta (−range_limit → +range_limit).
    device : torch.device, optional
        Device to use. Defaults to GPU if available.
    samples_per_eval : int, default=20
        Number of samples to reuse for evaluation.
    use_amp : bool, default=True
        Use automatic mixed precision on CUDA.
    compile_model : bool, default=True
        Try to compile model for faster inference.
    show_plot : bool, default=True
        If True, shows a 3D Plotly surface.

    Returns
    -------
    loss_surface : torch.Tensor
        2D tensor (grid_size × grid_size) of average losses.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Vectorize base parameters
    base_vec = (
        parameters_to_vector(model.parameters()).detach().to(device)
        if base_params is None
        else base_params.detach().to(device)
    )
    direction1 = direction1.detach().to(device, dtype=base_vec.dtype)
    direction2 = direction2.detach().to(device, dtype=base_vec.dtype)

    alpha_range = torch.linspace(-range_limit, range_limit, grid_size, device=device, dtype=base_vec.dtype)
    beta_range = alpha_range

    loss_surface = torch.empty((grid_size, grid_size), dtype=torch.float32, device="cpu")

    # Reuse a small evaluation batch
    eval_imgs, eval_labels = [], []
    total = 0
    for imgs, labels in test_loader:
        eval_imgs.append(imgs)
        eval_labels.append(labels)
        total += imgs.size(0)
        if total >= samples_per_eval:
            break

    eval_imgs = torch.cat(eval_imgs, 0)[:samples_per_eval].to(device, non_blocking=True)
    eval_labels = torch.cat(eval_labels, 0)[:samples_per_eval].to(device, non_blocking=True)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (device.type == "cuda" and use_amp)
        else nullcontext()
    )

    with amp_ctx:
        for i, alpha in enumerate(alpha_range):
            base_plus_alpha = base_vec + alpha * direction1
            for j, beta in enumerate(beta_range):
                vec = base_plus_alpha + beta * direction2
                vector_to_parameters(vec, model.parameters())

                outputs = model(eval_imgs).squeeze(1)
                loss = criterion(outputs, eval_labels)
                if hasattr(loss, "mean"):
                    loss = loss.mean()

                loss_surface[i, j] = float(loss)

    # Restore original params
    vector_to_parameters(base_vec, model.parameters())

    # Optional: plot
    if show_plot:
        X, Y = np.meshgrid(alpha_range.cpu().numpy(), beta_range.cpu().numpy())
        fig = go.Figure()

        fig.add_trace(go.Surface(x=X, y=Y, z=loss_surface.numpy(), colorscale="Viridis"))

        # Add red marker at origin (the base model)
        origin_z = float(loss_surface[grid_size // 2, grid_size // 2].item())
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[origin_z],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Model Parameters",
            )
        )

        fig.update_layout(
            title="Loss Landscape",
            scene=dict(
                xaxis_title="Direction 1",
                yaxis_title="Direction 2",
                zaxis_title="Loss",
            ),
            width=800,
            height=800,
        )
        fig.show()

    return loss_surface.cpu()
