#!/usr/bin/env python3
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

import pandas as pd
import sys

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import zeros
from numpy import zeros_like
from numpy import concatenate
from numpy import sqrt
from numpy import tanh
from numpy import fmin
from numpy import fmax
from numpy import exp
from numpy import log

import matplotlib.pyplot as plt

from matplotlib.cm import copper
from matplotlib.cm import plasma
from matplotlib.cm import viridis
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

from matplotlib.patches import FancyArrow

from matplotlib.colors import Colormap
from matplotlib.colors import Normalize

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho

from variable_conversions import rho
from variable_conversions import HBARC
from variable_conversions import milne_T
from variable_conversions import milne_mu
from variable_conversions import milne_entropy
from typing import List
from typing import Union
from typing import Tuple
from typing import Optional

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
    return newton(
        lambda tau:
            e_freezeout - e_interp(rho(tau, r, q)) / tau ** 4,
        x0=0.1,
        x1=0.2,
    )


def find_isentropic_temperature(
    mu: float,
    s_n: float,
) -> float:
    return newton(
        lambda t: 4.0 / tanh(mu / t) - mu / t - s_n,
        0.1
    )


def denergy_dtau(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
        consts: dict
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
    return_value = derivative * denergy_drho(ys, rho(tau, r, q), **consts) * tau
    return_value -= 4.0 * \
        energy(temperature=temperature, chem_potential=chem_potential,**consts)
    return return_value / tau ** 5


def denergy_dr(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
        consts: dict
) -> float:
    derivative = - q * r / tau
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return derivative * denergy_drho(ys, rho(tau, r, q), **consts) / tau ** 4


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
        T0: float,
        mu0_T0_ratios: ndarray,
        pi0: float,
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

    cfg = Config()

    ceos_temp_0 = cfg.ceos_temp_0
    ceos_mu_0 = array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0])
    consts = {'temperature_0': ceos_temp_0, 'chem_potential_0': ceos_mu_0}

    for k, alpha in enumerate(mu0_T0_ratios):
        y0s = [T0, alpha[0] * T0, alpha[1] * T0, alpha[2] * T0, pi0]
        soln_1 = odeint(eom, y0s, rhos_1, args=(ceos_temp_0, ceos_mu_0,))
        soln_2 = odeint(eom, y0s, rhos_2, args=(ceos_temp_0, ceos_mu_0,))
        t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = [concatenate((soln_1[:, ii][::-1], soln_2[:, ii]))
                    for ii in [1,2,3]]
        pi_hat = concatenate((soln_1[:, 4][::-1], soln_2[:, 4]))
        rhos = concatenate((rhos_1[::-1], rhos_2))

        t_interp = interp1d(rhos, t_hat)
        mu_interp = [interp1d(rhos, f) for f in mu_hat]
        pi_interp = interp1d(rhos, pi_hat)

        e_interp = interp1d(rhos, [energy(*x, **consts) for x in
                                   zip(t_hat,zip(*mu_hat))])

        freezeout_times = zeros((xs.size, 2))

        for i, x in enumerate(xs):
            try:
                freezeout_times[i] = [
                    x,
                    find_freezeout_tau(
                        e_interp, e_freezeout, x, q
                    )
                ]
            except ValueError:
                freezeout_times[i] = [x, 1.e-12]
        list_FO_surfaces.append(freezeout_times)
        min_tau = min(fmin(freezeout_times[:, 1], min_tau))
        max_tau = max(fmax(freezeout_times[:, 1], max_tau))

        # some magic to make colorbars happen
#        try:
#            freezeout_s = milne_entropy(
#                tau=freezeout_times[:, 1],
#                x=freezeout_times[:, 0],
#                y=0.0,
#                q=1.0,
#                ads_T=t_interp,
#                ads_mu=mu_interp,
#            )
#            list_FO_entropies.append(freezeout_s)
#            min_s = min(fmin(freezeout_s, min_s))
#            max_s = max(fmax(freezeout_s, max_s))
#        except ValueError:
#            list_FO_entropies.append(0.0)

        normal_vectors = zeros((xs.size // skip_size, 2))
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            _rho = rho(tau=tau_FO, r=x, q=q)
            try:
                normal_vectors[i] = [
                    -denergy_dr(
                        ys=array([
                            t_interp(_rho),
                            mu_interp[0](_rho),
                            mu_interp[1](_rho),
                            mu_interp[2](_rho),
                            pi_interp(_rho)
                        ]),
                        tau=tau_FO,
                        r=x,
                        q=q,
                        consts=consts
                    ),
                    -denergy_dtau(
                        ys=array([
                            t_interp(_rho),
                            mu_interp[0](_rho),
                            mu_interp[1](_rho),
                            mu_interp[2](_rho),
                            pi_interp(_rho)
                        ]),
                        tau=tau_FO,
                        r=x,
                        q=q,
                        consts=consts
                    ),
                ]

                norm = sqrt(abs(
                    normal_vectors[i, 0] ** 2 - normal_vectors[i, 1] ** 2
                ))
                normal_vectors[i] = norm_scale * normal_vectors[i] / norm
            except ValueError as e:
                print(e.with_traceback("Warning: Value error in normal vector calculation"))
                print("x= ", x, " tau_FO= ", tau_FO, " rho= ", _rho)

                normal_vectors[i] = [0.0,0.0]
                norm = 1.0

        list_FO_normals.append(normal_vectors)

    return min_s, max_s, min_tau, max_tau, list_FO_surfaces, \
        list_FO_entropies, list_FO_normals


def dx_dtau(
        ys: ndarray,
        tau: Union[float, ndarray],
        q: float
) -> Union[float, ndarray]:
    x, y, eta = ys
    r2 = x ** 2 + y ** 2
    v_r = 2 * q * tau / (1 + q ** 2 * (r2 + tau ** 2))
    r = sqrt(r2)
    return array([x / r, y / y, 0]) * v_r

def load_sim_data():
    sim_result_path = sys.argv[1]
    col_names=['divEener','divE_x','divE_y', 'gsub', "uout_x", "uout_y",
                "swsub", "bulksub", "pit_t", "pi_xx", "pi_yy", "pi_zz", "pi_xy",
                "t", "x", "y", "s", "e", "T", "muB", "muS", "muQ", "w","cs2"]

    df = pd.read_table(sim_result_path,
                        names=col_names,sep=' ',header=0).query('abs(x-y) < 1.E-3')
    df['r'] = sqrt(df['x']**2 + df['y']**2)
    df['divE_r'] = sqrt(df['divE_x']**2 + df['divE_y']**2)
#    df['norm'] = sqrt(-df['divE_r']**2+df['divEener']**2)
#    df['divE_r'] = df['divE_r']/df['norm']
#    df['divE_x'] = df['divE_x']/df['norm']
#    df['divE_y'] = df['divE_y']/df['norm']
#    df['divEener'] = df['divEener']/df['norm']
    return df

def solve_and_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        y0s: ndarray,
        mu0_T0_ratios: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        e_freezeout: float,
        q: float,
        # colors: List[str],
        add_labels: bool = True,
        norm_scale: float = 0.1,
        heat_map: Colormap = None,
        update_color_bar: bool = False,
        plot_s_n: bool = False,
) -> None:

    sim_data = load_sim_data()
    xs = sim_data['r'].unique()
    skip_size = 1
    min_s, max_s, min_tau, max_tau, fo_surfaces, \
        fo_entropies, fo_normals = do_freezeout_surfaces(
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            T0=y0s[0],
            mu0_T0_ratios=mu0_T0_ratios,
            pi0=y0s[2],
            skip_size=skip_size,
            e_freezeout=e_freezeout,
            norm_scale=norm_scale,
            q=q,
        )

    evol_taus_log = linspace(log(0.01), log(10), 1000)
    evol_taus = exp(evol_taus_log)

    xis = mu0_T0_ratios
    for itr in range(len(mu0_T0_ratios)):
        try:
            freezeout_times = fo_surfaces[itr]
        except IndexError:
            print("Out of bounds for array freezeout_times. Iteration: ", itr)
            print(mu0_T0_ratios.size)
            sys.exit(1)()
        #freezeout_s = fo_entropies[itr]
        normal_vectors = fo_normals[itr]

        arrows = zeros((xs.size // skip_size,), dtype=FancyArrow)
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            arrows[i] = ax[0].arrow(
                x=x,
                y=tau_FO,
                dx=normal_vectors[i, 0],
                dy=normal_vectors[i, 1],
                head_width=0.02,
                head_length=0.02,
                color='black',
                lw=2.0,
            )

        arrows_sim = zeros((xs.size // skip_size,), dtype=FancyArrow)
        arrows_sim = sim_data[['r', 't', 'divEener', 'divE_r']].to_dict('records')
        for d in arrows_sim:
            d['x'] = d.pop('r')
            d['y'] = d.pop('t')
            d['dy'] = d.pop('divEener')*norm_scale
            d['dx'] = d.pop('divE_r')*norm_scale
            arrows[i] = ax[0].arrow(**d,
                head_width=0.01,
                head_length=0.01,
                color='red',
                alpha=1.0,
                linewidth=2,
            )

       #heat_map = get_cmap(copper, freezeout_s.size)
        ax[0].scatter(
            sim_data['r'],
            sim_data['t'],
            s=10.0,
            c='red',
            label='CCAKE'
        )


        ax[0].scatter(
            freezeout_times[:, 0],
            freezeout_times[:, 1],
            s=10.0,
            c='black',
            label='Semi-Analytical'
        )
        ax[0].legend(frameon=False,fontsize='20')

    return


def main():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.0 * 7, 1 * 7))
    fig.patch.set_facecolor('white')

    y0s = array([.25 / HBARC  ,0.05 / HBARC,0.0])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 3, 1000)
    xs = linspace(0, 6, 1000)

    solve_and_plot(
        fig=fig,
        ax=[ax],
        y0s=y0s,
        mu0_T0_ratios=array([[.2,0,0],]),
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        e_freezeout=1.0 / HBARC,
        q=1,
        # colors=['blue'],
        update_color_bar=True,
        plot_s_n=True,
        norm_scale=0.10
    )

    costumize_axis(
        ax=ax,
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/$c$]',
    )
    ax.set_xlim(0, 2.25)
    ax.set_ylim(.9, 2.0)
    # ax[0].text(3.4, 0.55, r'$\mu_0/T_0=1$', fontsize=18)
    ax.text(0.1, 1.05, r'$\mu_0/T_0=0.2$', fontsize=18)
    ax.text(0.1, 0.95, "EoS 2", fontsize=18, bbox=dict(
            boxstyle='round', facecolor='white'))

    fig.tight_layout()
    fig.savefig('./freeze-out-surface-ccake.pdf')


if __name__ == "__main__":
    main()
