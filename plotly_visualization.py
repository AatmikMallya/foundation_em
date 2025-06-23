import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path

__all__ = ["plotly_visualize_reconstructions"]

def _volume_to_plotly_fig(volume: np.ndarray, title: str = "Volume", opacity: float = 0.15,
                           opacityscale = ((0,0), (0.3,0), (0.6,0.3), (1,1)),
                           colorscale: str = "Gray", surface_count: int = 8):
    """Return a Plotly figure that renders a 3-D volume with a transparency TF.

    Args
    ----
    volume : (D,H,W) numpy array, values assumed to be in [0,1] or arbitrary float range
    title  : str – title shown in the viewer
    opacity: base opacity multiplier (Plotly `opacity` arg)
    opacityscale: list/tuple   – custom alpha transfer-function; see plotly docs
    colorscale: str ‑ name of plotly colourscale
    surface_count: int – number of isosurfaces to draw (lower ⇒ faster)
    """
    D,H,W = volume.shape
    # Cartesian coordinates
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    flat_val = volume.flatten()

    fig = go.Figure(
        go.Volume(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=flat_val,
            opacity=opacity,
            opacityscale=opacityscale,
            surface_count=surface_count,
            colorscale=colorscale,
            showscale=False,
        )
    )
    fig.update_layout(title=title, scene=dict(aspectmode="cube"))
    return fig


def plotly_visualize_reconstructions(model, dataloader, device, step: int, mask_ratio: float,
                                     tag: str = "sample", num_examples: int = 2):
    """Generate interactive Plotly figures for a few examples and return them.

    Parameters
    ----------
    model : MAE model (with unpatchify method)
    dataloader : torch.utils.data.DataLoader – yields volumes
    device : torch.device
    step : int – global step (only used for titles)
    mask_ratio : float – current mask ratio (for info only)
    tag : str – label prefix
    num_examples : int – how many examples to render

    Returns
    -------
    list[plotly.graph_objects.Figure]
    """
    model.eval()
    figs = []
    with torch.no_grad():
        for batch in dataloader:
            if len(figs) >= num_examples:
                break
            vols = batch.to(device)
            loss, pred, mask, patch_stats = model(vols, mask_ratio=mask_ratio)
            pred_denorm = pred
            if patch_stats is not None:
                mean, var = patch_stats
                pred_denorm = pred_denorm * (var.add(1e-6).sqrt()) + mean
            recon = model.unpatchify(pred_denorm).cpu().numpy()
            orig = vols.cpu().numpy()
            B = orig.shape[0]
            for i in range(B):
                if len(figs) >= num_examples:
                    break
                orig_vol = orig[i, 0]
                recon_vol = recon[i, 0]
                # Normalise to 0-1 for rendering
                def _norm(v):
                    vmin, vmax = v.min(), v.max()
                    if vmax > vmin:
                        return (v - vmin) / (vmax - vmin)
                    return np.zeros_like(v)
                # Invert intensities so low values (membrane) are opaque & high (cytosol) transparent
                o_norm = 1.0 - _norm(orig_vol)
                r_norm = 1.0 - _norm(recon_vol)

                fig_orig = _volume_to_plotly_fig(o_norm, title=f"{tag}-orig step{step}")
                fig_recon = _volume_to_plotly_fig(r_norm, title=f"{tag}-recon step{step}", colorscale="Viridis")
                figs.extend([fig_orig, fig_recon])
    return figs 