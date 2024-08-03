import os, math, datetime

import numpy as np
import xarray, matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from typing import Optional

def scale(data: xarray.Dataset, center: Optional[float] = None, robust: bool = False) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))

    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    
    return (data, matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    output_dir: str = "/scratch/ll44/sc6160/out/plots",
    output_prefix: str = ""):

    os.makedirs(output_dir, exist_ok=True)

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,  plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)

    def update(frame):
        
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        # for im, (plot_data, norm, cmap) in zip(images, data.values()):
        #     im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))
        for idx, (plot_data, _, _) in enumerate(data.values()):
           
           im = images[idx]
           im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

           # save frame
           plt.savefig(os.path.join(output_dir, f"{output_prefix}frame_{frame:02d}.png"))

    anim = animation.FuncAnimation(fig=figure, func=update, frames=max_steps, interval=250)
    anim.save(os.path.join(output_dir, f"{output_prefix}animation.gif"), writer="pillow", fps=2)

    plt.close(figure.number)