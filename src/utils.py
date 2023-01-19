import matplotlib.pyplot as plt
import matplotlib as ml
import numpy as np
from scipy.interpolate import interp1d


def at_wavenumbers(
    from_wavenumbers: np.ndarray,
    to_wavenumbers: np.ndarray,
    spectra: np.ndarray,
    extrapolation_mode=None,
    extrapolation_value: int = 0,
) -> np.ndarray:
    """
    Interpolates spectrum at another wavenumbers
    :param from_wavenumbers: initial wavenumbers
    :param to_wavenumbers: to what wavenumbers interpolate
    :param spectra: spectra
    :param extrapolation_mode: 'constant' or 'boundary' or None (then raise error)
    :param extrapolation_value: value to which interpolate in case of
    constant extrapolation
    """
    to_max = to_wavenumbers.max()
    to_min = to_wavenumbers.min()
    if to_max > from_wavenumbers.max():
        if extrapolation_mode is None:
            raise ValueError("Range of to_wavenumbers exceeds boundaries of from_wavenumbers")
        from_wavenumbers = np.append(from_wavenumbers, to_max)
        if extrapolation_mode == "constant":
            spectra = np.append(spectra, [extrapolation_value], axis=-1)
        elif extrapolation_mode == "boundary":
            spectra = np.append(spectra, [spectra[..., -1]], axis=-1)
        else:
            raise ValueError(f"Unknown extrapolation_mode {extrapolation_mode}")
    if to_min < from_wavenumbers.min():
        if extrapolation_mode is None:
            raise ValueError("Range of to_wavenumbers exceeds boundaries of from_wavenumbers")
        from_wavenumbers = np.insert(from_wavenumbers, 0, to_min)
        if extrapolation_mode == "constant":
            spectra = np.insert(spectra, 0, [extrapolation_value], axis=-1)
        elif extrapolation_mode == "boundary":
            spectra = np.insert(spectra, 0, [spectra[..., 0]], axis=-1)
        else:
            raise ValueError(f"Unknown extrapolation_mode {extrapolation_mode}")

    return interp1d(from_wavenumbers, spectra)(to_wavenumbers)


def set_spectral_plot_context(
    fig=None,
    ax=None,
    xlabel=r"$\tilde{\nu}\;\; (cm^{-1})$",
    ylabel="A",
    rotation_y=0,
    xlim=None,
    ylim=None,
    label_fs=34,
    tick_pad=6
):
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    if ax is None:
        ax = plt.gca()
    ml.rcParams['font.family'] = 'sans-serif'

    fig.frameon = True
    fig.edgecolor = "red"
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, rotation=rotation_y, fontsize=label_fs, labelpad=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both' , direction='in', width=3, length=7)
    ax.invert_xaxis()
    ax.grid(axis="y", which="major", linestyle="--", lw=1.5, alpha=0.6)
    ax.grid(axis="y", which="minor", linestyle="--", lw=1., alpha=0.6)
    ax.tick_params(axis='both', which='major', pad=tick_pad)
    ax.yaxis.set_major_formatter(ml.ticker.FormatStrFormatter('%.2f'))

def set_spectral_legend(S, spec, loc="upper center", ncol=2):
    pl_max, pl_min = max(spec.max(), S.max()), min(spec.min(), S.min())
    plt.ylim(pl_min - (pl_max - pl_min)*0.09, pl_max + (pl_max - pl_min)*0.225)
    leg = plt.legend(fontsize=28, 
                     frameon=True, 
                     facecolor='white',
                     edgecolor='1.0',
                     ncol=ncol, 
                     handlelength=1., 
                     handletextpad=0.4,
                     labelspacing=.15,
                     columnspacing=1.25,
                     loc=loc)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
