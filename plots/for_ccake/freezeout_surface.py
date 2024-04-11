from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import zeros
from numpy import concatenate
from numpy import sqrt
from numpy import fmin
from numpy import fmax

import matplotlib.pyplot as plt

from matplotlib.cm import copper
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

from matplotlib.patches import FancyArrow

from matplotlib.colors import Normalize

from typing import List
from typing import Union
from typing import Tuple
from typing import Optional

import sys; sys.path.append('../')

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho

from variable_conversions import rho
from variable_conversions import milne_entropy
from variable_conversions import HBARC

CONST_T0 = 1.0
CONST_MU0 = array([1.0, 1.0, 1.0]).reshape(-1, 1)

class Config:
    def __init__(self):
        self.tau_0: Optional[float] = None
        self.tau_f: Optional[float] = None
        self.tau_step: Optional[float] = None
        self.temp_0: Optional[float] = None
        self.muB_0: Optional[float] = None
        self.muS_0: Optional[float] = None
        self.muQ_0: Optional[float] = None
        self.ceos_temp_0: Optional[float] = None
        self.ceos_muB_0: Optional[float] = None
        self.ceos_muS_0: Optional[float] = None
        self.ceos_muQ_0: Optional[float] = None
        self.pi_0: Optional[float] = None
        self.tol: Optional[float] = None
        self.output_dir: Optional[str] = None

        self.read_from_config()

    def read_from_config(self):
        with open('run.cfg', 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Initialization stuff
                key, value = line.split()[:2]
                print(key, value)
                if key == 'tau_0':
                    self.tau_0 = float(value)
                elif key == 'tau_f':
                    self.tau_f = float(value)
                elif key == 'tau_step':
                    self.tau_step = float(value)
                elif key == 'temp_0':
                    self.temp_0 = float(value)
                elif key == 'muB_0':
                    self.muB_0 = float(value)
                    if self.muB_0 == 0:
                        self.muB_0 = 1e-20
                elif key == 'muS_0':
                    self.muS_0 = float(value)
                    if self.muS_0 == 0:
                        self.muS_0 = 1e-20
                elif key == 'muQ_0':
                    self.muQ_0 = float(value)
                    if self.muQ_0 == 0:
                        self.muQ_0 = 1e-20
                elif key == 'pi_0':
                    self.pi_0 = float(value)
                # EOS stuff
                elif key == 'ceos_temp_0':
                    self.ceos_temp_0 = float(value)
                elif key == 'ceos_muB_0':
                    self.ceos_muB_0 = float(value)
                elif key == 'ceos_muS_0':
                    self.ceos_muS_0 = float(value)
                elif key == 'ceos_muQ_0':
                    self.ceos_muQ_0 = float(value)
                # Utility
                elif key == 'tolerance':
                    self.tol = float(value)
                elif key == 'output_dir':
                    self.output_dir = value


def find_freezeout_tau(
        e_interp: interp1d,
        e_freezeout: float,
        r: float,
        q: float,
) -> float:
    def f(tau: float) -> float:
        rh = rho(tau, r, q)
        return e_freezeout - e_interp(rh) / tau ** 4

    try:
        value = newton(
            f,
            x0=0.1,
            x1=0.2,
        )
    except (RuntimeError, ValueError):
        try:
            value = newton(
                f,
                x0=0.01,
                x1=0.02,
            )
        except (RuntimeError, ValueError):
            value = newton(
                f,
                x0=0.001,
                x1=0.002,
            )
    return value


def find_isentropic_temperature(
    mu: ndarray,
    s_n: float,
) -> float:
    return newton(
        lambda t: (CONST_MU0[0, 0] / CONST_T0) ** 2 * t / mu - s_n,
        x0=0.1,
        x1=0.2
    )


def denergy_dtau(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    temperature, mu_B, mu_S, mu_Q, _ = ys
    chem_potential = array([mu_B, mu_S, mu_Q])
    derivative = 1 + q ** 2 * (r ** 2 + tau ** 2)
    derivative /= tau * sqrt(
        1
        +
        q ** 4 * (r ** 2 - tau ** 2) ** 2
        +
        2 * q ** 2 * (r ** 2 + tau ** 2)
    )
    return_value = derivative * denergy_drho(
        ys, rho(tau, r, q), temperature_0, chem_potential_0) * tau
    return_value -= 4.0 * energy(
        temperature,
        chem_potential,
        temperature_0,
        chem_potential_0
    )
    return return_value / tau ** 5


def denergy_dr(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    derivative = - q * r / tau
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return derivative * denergy_drho(
        ys, rho(tau, r, q), temperature_0, chem_potential_0) / tau ** 4


def find_color_indices(
        max_s: float,
        min_s: float,
        num_colors: int,
        s: Union[float, ndarray],
) -> Union[float, ndarray]:
    ds = (max_s - min_s) / num_colors
    return (s - min_s) // ds


def do_freezeout_surfaces(
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        y0s: ndarray,
        e_freezeout: float,
        skip_size: int,
        norm_scale: float,
        q: float
) -> Tuple[
    float,
    float,
    float,
    float,
    List[ndarray],
    List[ndarray],
    List[ndarray]
]:
    # This function does a lot more work than it needs to...
    # returns:
    #   - min entropy at freezeout for all ICs
    #   - max entropy at freezeout for al ICs
    #   - min freeze-out time for all ICs
    #   - max freeze-out time for all ICs
    #   - list of freeze-out surface (r_FO, tau_FO) arrays for ICs
    #   - list of entropy densities of freeze-out surface for all ICs
    #   - list of normalized normal vectors for freeze-out surfaces

    list_FO_surfaces = []
    list_FO_entropies = []
    list_FO_normals = []
    min_s = 1e99
    max_s = -1e99
    min_tau = 1e99
    max_tau = -1e99

    ics = {'temperature_0': CONST_T0, 'chem_potential_0': CONST_MU0}
    soln_1 = odeint(eom, y0s, rhos_1, args=(CONST_T0, CONST_MU0,))
    soln_2 = odeint(eom, y0s, rhos_2, args=(CONST_T0, CONST_MU0,))
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = [concatenate((soln_1[:, i][::-1], soln_2[:, i]))
                for i in [1, 2, 3]]
    pi_hat = concatenate((soln_1[:, 4][::-1], soln_2[:, 4]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = [interp1d(rhos, f) for f in mu_hat]
    pi_interp = interp1d(rhos, pi_hat)

    e_interp = interp1d(rhos, energy(t_hat, mu_hat, **ics))

    freezeout_times = zeros((xs.size, 2))

    for i, x in enumerate(xs):
        try:
            freezeout_times[i] = [
                x,
                find_freezeout_tau(
                    e_interp, e_freezeout, x, q
                )
            ]
        except (RuntimeError or ValueError):
            # print(i, x, "failed")
            freezeout_times[i] = [x, 1e-12]

    list_FO_surfaces.append(freezeout_times)
    min_tau = min(fmin(freezeout_times[:, 1], min_tau))
    max_tau = max(fmax(freezeout_times[:, 1], max_tau))

    # some magic to make colorbars happen
    freezeout_s = milne_entropy(
        tau=freezeout_times[:, 1],
        x=freezeout_times[:, 0],
        y=0.0,
        q=1.0,
        ads_T=t_interp,
        ads_mu=mu_interp,
        **ics,
    )

    list_FO_entropies.append(freezeout_s)
    min_s = min(fmin(freezeout_s, min_s))
    max_s = max(fmax(freezeout_s, max_s))

    normal_vectors = zeros((xs.size // skip_size, 2))
    for i, var in enumerate(freezeout_times[::skip_size]):
        x, tau_FO = var
        _rho = rho(tau=tau_FO, r=x, q=q)
        normal_vectors[i] = [
            -denergy_dr(
                ys=array([
                    t_interp(_rho),
                    *[f(_rho) for f in mu_interp],
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q,
                **ics
            ),
            -denergy_dtau(
                ys=array([
                    t_interp(_rho),
                    *[f(_rho) for f in mu_interp],
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q,
                **ics
            ),
        ]

        norm = sqrt(abs(
            normal_vectors[i, 0] ** 2 - normal_vectors[i, 1] ** 2
        ))
        normal_vectors[i] = norm_scale * normal_vectors[i] / norm
    list_FO_normals.append(normal_vectors)

    return min_s, max_s, min_tau, max_tau, list_FO_surfaces, \
        list_FO_entropies, list_FO_normals


def solve_and_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        e_freezeout: float,
        q: float,
        norm_scale: float = 0.1,
) -> None:

    skip_size = 4
    min_s, max_s, min_tau, max_tau, fo_surfaces, \
        fo_entropies, fo_normals = do_freezeout_surfaces(
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            y0s=y0s,
            skip_size=skip_size,
            e_freezeout=e_freezeout,
            norm_scale=norm_scale,
            q=q,
        )

    for itr in range(len(fo_surfaces)):
        freezeout_times = fo_surfaces[itr]
        freezeout_s = fo_entropies[itr]
        normal_vectors = fo_normals[itr]

        arrows = zeros((xs.size // skip_size,), dtype=FancyArrow)
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            arrows[i] = ax.arrow(
                x=x,
                y=tau_FO,
                dx=normal_vectors[i, 0],
                dy=normal_vectors[i, 1],
                head_width=0.01,
                head_length=0.01,
                color='black'
            )

        heat_map = get_cmap(copper, freezeout_s.size)

        ax.scatter(
            freezeout_times[:, 0],
            freezeout_times[:, 1],
            c=find_color_indices(
                min_s=min_s,
                max_s=max_s,
                num_colors=freezeout_s.size,
                s=freezeout_s,
            ),
            s=3.0,
            cmap=heat_map,
            norm=Normalize(vmin=0, vmax=freezeout_s.size)
        )

        if itr == 0:
            norm = Normalize(
                vmin=min_s,
                vmax=max_s,
            )
            s = ScalarMappable(
                norm=norm,
                cmap=heat_map
            )
            cax = fig.colorbar(s, ax=ax, orientation='vertical', pad=0.01,
                            format='%.2f').ax
            cax.yaxis.set_ticks(linspace(min_s, max_s, 7))
            for t in cax.get_yticklabels():
                t.set_fontsize(18)
            cax.set_ylabel(r'$s(\tau, x)$ [GeV$^{3}$]', fontsize=20)

    return


def main(y0s: ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.0 * 7, 1 * 7))
    fig.patch.set_facecolor('white')

    rhos_1 = linspace(-40, 0, 4000)[::-1]
    rhos_2 = linspace(0, 40, 4000)
    xs = linspace(0, 6, 1000)

    solve_and_plot(
        fig=fig,
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0,
        q=1,
        norm_scale=0.05
    )

    costumize_axis(
        ax=ax,
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/$c$]',
    )
    # ax.set_xlim(0, 6.0)
    # ax.set_ylim(0.0, 3.0)

    fig.tight_layout()
    fig.savefig('./freezeout-surface.pdf')


if __name__ == "__main__":
    cfg = Config()
    
    temp_0 = cfg.temp_0 * cfg.tau_0 / HBARC
    muB_0 = cfg.muB_0 * cfg.tau_0 / HBARC
    muS_0 = cfg.muS_0 * cfg.tau_0 / HBARC
    muQ_0 = cfg.muQ_0 * cfg.tau_0 / HBARC
    y0s = array([temp_0, muB_0, muS_0, muQ_0, cfg.pi_0])

    CONST_MU0 = cfg.ceos_temp_0
    CONST_MU0 = array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0]).reshape(-1, 1)

    main(y0s)
