from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import newton

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
from numpy import sum as npsum
from numpy import isnan

import matplotlib.pyplot as plt

from matplotlib.cm import copper
from matplotlib.cm import plasma
from matplotlib.cm import viridis
from matplotlib.cm import coolwarm
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable

from matplotlib.patches import FancyArrow

from matplotlib.colors import Colormap
from matplotlib.colors import Normalize

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho
from equations_of_motion import entropy
from equations_of_motion import number

from variable_conversions import rho
from variable_conversions import HBARC
from variable_conversions import milne_T
from variable_conversions import milne_mu
from variable_conversions import milne_entropy

from typing import List
from typing import Union
from typing import Tuple

CONST_T0 = 1.0
CONST_MU0 = array([1.0, 1.0, 1.0]).reshape(-1, 1)


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


# TODO: cccake eos
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


# TODO: cccake eos
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
        T0: float,
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

    for k, alpha in enumerate([1e-20, 1.5, 3]):
        y0s = [T0, alpha * T0, alpha * T0, alpha * T0, pi0]
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


def solve_and_plot(
        fig: plt.Figure,
        ax: plt.Axes,
        y0s: ndarray,
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

    skip_size = 8
    min_s, max_s, min_tau, max_tau, fo_surfaces, \
        fo_entropies, fo_normals = do_freezeout_surfaces(
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            T0=y0s[0],
            pi0=y0s[2],
            skip_size=skip_size,
            e_freezeout=e_freezeout,
            norm_scale=norm_scale,
            q=q,
        )

    evol_taus_log = linspace(log(0.01), log(3), 1000)
    evol_taus = exp(evol_taus_log)

    xis = [1e-20, 1, 2, 3]
    shifts = array([0.0, 0.0, 1.0, 2.0]) / 100
    colors = ['black', 'red', 'blue']
    linestyles = ['solid', 'dashed', 'dotted']
    for itr in range(len(fo_surfaces)):
        freezeout_times = fo_surfaces[itr]
        freezeout_s = fo_entropies[itr]
        normal_vectors = fo_normals[itr]

        arrows = zeros((xs.size // skip_size,), dtype=FancyArrow)
        for i, var in enumerate(freezeout_times[::skip_size]):
            x, tau_FO = var
            arrows[i] = ax[0].arrow(
                x=x,
                y=tau_FO,
                dx=normal_vectors[i, 0],
                dy=normal_vectors[i, 1],
                head_width=0.01,
                head_length=0.01,
                color='black'
            )

        heat_map = get_cmap(copper, freezeout_s.size)

        ax[0].scatter(
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

        if update_color_bar and itr == 1:
            norm = Normalize(
                vmin=min_s,
                vmax=max_s,
            )
            s = ScalarMappable(
                norm=norm,
                cmap=heat_map
            )
            cax = fig.colorbar(s, ax=ax[0], orientation='vertical', pad=0.01,
                               format='%.2f').ax
            cax.yaxis.set_ticks(linspace(min_s, max_s, 7))
            for t in cax.get_yticklabels():
                t.set_fontsize(18)
            cax.set_ylabel(r'$s(\tau, x)$ [GeV$^{3}$]', fontsize=20)

        if itr == 0:
            continue

        x_FOs = freezeout_times[:, 0]
        tau_FOs = freezeout_times[:, 1]

        # taus = tau_FOs
        # cmap = get_cmap(plasma, taus.size)
        # if update_color_bar and itr == 1:
        #     tau0 = evol_taus[0]
        #     tauf = evol_taus[-1]
        #     norm = Normalize(vmin=tau0, vmax=tauf)
        #     sm = ScalarMappable(norm=norm, cmap=cmap)
        #     cax = fig.colorbar(sm, ax=ax[1], orientation='vertical', pad=0.01,
        #                     format='%.2f').ax
        #     cax.yaxis.set_ticks(linspace(tau0, tauf, 7))
        #     for t in cax.get_yticklabels():
        #         t.set_fontsize(18)
        #     cax.set_ylabel(r'$\tau$ [fm/$c$]', fontsize=20)

        xi = xis[itr]
        ys = [y0s[0], xi * y0s[0], xi * y0s[0], xi * y0s[0], y0s[2]]
        soln_1 = odeint(eom, ys, rhos_1, args=(CONST_T0, CONST_MU0))
        soln_2 = odeint(eom, ys, rhos_2, args=(CONST_T0, CONST_MU0))
        rhos = concatenate((rhos_1[::-1], rhos_2))
        t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
        mu_hat = [concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
                  for i in [1, 2, 3]]
        t_interp = interp1d(rhos, t_hat)
        mu_interp = [interp1d(rhos, f) for f in mu_hat]

        s_interp = interp1d(rhos, entropy(t_hat, mu_hat))
        n_interp = [*interp1d(rhos, number(t_hat, mu_hat))]

        rs = linspace(0.01, 1.0, 100)
        evol_mus = zeros((rs.size, evol_taus.size))
        evol_temps = zeros_like(evol_mus)
        for nn, r0 in enumerate(rs):
            evol_xs = odeint(dx_dtau, array(
                [r0, r0, 0]), exp(evol_taus_log), args=(1.0,))
            evol_rs = sqrt(evol_xs[:, 0] ** 2 + evol_xs[:, 1] ** 2)
            evol_mus[nn] = milne_mu(evol_taus, evol_rs, 1.0, mu_interp[0])
            evol_temps[nn] = milne_T(evol_taus, evol_rs, 1.0, t_interp)

        ax[1].hist2d(
            evol_mus.reshape(-1,),
            evol_temps.reshape(-1,),
            bins=100,
            cmap=[viridis, plasma][itr - 1],
            norm='log',
            alpha=1.0 / itr
        )

        # ax[1].scatter(
        #     # milne_mu(taus, 0.0, 1.0, mu_interp),
        #     # milne_T(taus, 0.0, 1.0, t_interp),
        #     milne_mu(taus, x_FOs, 1.0, mu_interp),
        #     milne_T(taus, x_FOs, 1.0, t_interp) + shifts[itr],
        #     color='black',
        #     s=4.0,
        # )

        # s_n = milne_entropy(taus, x_FOs, 1.0, t_interp, mu_interp=)
        # s_n = milne
        # mus = milne_mu(taus, x_FOs, 1.0, mu_interp)
        # ts = array([
        #     find_isentropic_temperature(
        #         mu=mu,
        #         s_n=s_n
        #     )
        #     for mu in mus
        # ])
        # ax[1].scatter(
        #     mus,
        #     ts,
        #     c=taus,
        #     s=2.0,
        #     cmap=cmap
        # )

        if plot_s_n:
            _xs = concatenate((-xs[::-1], xs))
            for k, tau in enumerate([1.2, 2.0, 3.0]):
                label_string_2 = ''
                if k == 0:
                    label_string_2 += r'$\mu_0/T_0 = ' + f'{xis[itr]:.1f}$'

                rh = rho(tau, _xs, 1)
                ax[2].plot(
                    _xs,
                    s_interp(rh) / n_interp(rh)[0],
                    ls=linestyles[k],
                    color=colors[itr],
                    label=label_string_2
                )
    return heat_map


# Plot the tau components and r components separately, to see which has
# the large change


def main():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3.0 * 7, 1 * 7))
    fig.patch.set_facecolor('white')

    rhos_1 = linspace(-30, 0, 1000)[::-1]
    rhos_2 = linspace(0, 30, 1000)
    xs = linspace(0, 6, 1000)

    t0 = 1.2
    for alpha in [0.01, 0.03, 0.05]:
        print(alpha)
        y0s = array([
            t0,
            *[alpha * t0 for _ in range(3)],
            0.0
        ])

        solve_and_plot(
            fig=fig,
            ax=ax,
            y0s=y0s,
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            e_freezeout=0.1 / HBARC,
            q=1,
            # colors=['blue'],
            update_color_bar=True,
            plot_s_n=True,
            norm_scale=0.05
        )

    costumize_axis(
        ax=ax[0],
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/$c$]',
    )
    ax[0].set_xlim(0, 6.0)
    ax[0].set_ylim(0.0, 3.0)
    ax[0].text(4.5, 1.6, r'$\mu_0/T_0=3$', fontsize=18)
    ax[0].text(4.5, 0.7, r'$\mu_0/T_0=2$', fontsize=18)
    # ax[0].text(3.4, 0.55, r'$\mu_0/T_0=1$', fontsize=18)
    ax[0].text(3.0, 0.1, r'$\mu_0/T_0=0$', fontsize=18)

    costumize_axis(
        ax=ax[1],
        x_title=r'$\mu$ [GeV]',
        y_title=r'$T$ [GeV]'
    )
    ax[1].axhline(0.2, color='black')
    ax[1].text(1.0, 0.21, '$T=200$ MeV', fontsize=18)
    ax[1].set_ylim(top=2)
    # ax[1].set_yscale('log')
    # ax[1].text(0.1, 0.7, r'$\mu_0/T_0=1$', fontsize=18)
    # ax[1].text(0.65, 0.7, r'$\mu_0/T_0=2$', fontsize=18)
    # ax[1].text(1.05, 0.7, r'$\mu_0/T_0=3$', fontsize=18)

    # ax[1].text(0.01, 0.159, r'$+0.01$', fontsize=16)
    # ax[1].text(0.01, 0.169, r'$+0.02$', fontsize=16)

    # ax[2].set_aspect(1.0, anchor='SW')
    costumize_axis(
        ax=ax[2],
        x_title=r'$x$ [fm]',
        y_title=r'$s(\tau, x)/n(\tau, x)$'
    )
    ax[2].legend(loc='upper center', fontsize=20)

    fig.tight_layout()
    fig.savefig('./freeze-out-surface.pdf')


if __name__ == "__main__":
    main()
