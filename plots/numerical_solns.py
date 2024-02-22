from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import concatenate

import matplotlib.pyplot as plt

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import number
from equations_of_motion import entropy

from variable_conversions import milne_T
from variable_conversions import milne_mu
from variable_conversions import milne_pi
from variable_conversions import rho
from variable_conversions import HBARC

from typing import List

T_PLOT = (0, 0)
MU_PLOT = (0, 1,)
PIXX_PLOT = (1, 0)
PIXY_PLOT = (1, 1)

E_PLOT = (0)
N_PLOT = (1,)
S_PLOT = (2,)


def solve_and_plot(
        ax_1: plt.Axes,
        ax_2: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        taus: ndarray,
        color: List[str],
        linestyle: List[str],
        second_label: str,
        add_labels: bool = False,
) -> None:
    soln_1 = odeint(eom, y0s, rhos_1)
    soln_2 = odeint(eom, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)

    for n, tau in enumerate(taus):
        t_evol = milne_T(tau, xs, 1, t_interp)
        mu_evol = milne_mu(tau, xs, 1, mu_interp)

        e_evol = energy(t_evol, mu_evol)  # / HBARC ** 3
        n_evol = number(t_evol, mu_evol)  # / HBARC ** 3
        s_evol = entropy(t_evol, mu_evol)  # / HBARC ** 3

        ax_1[T_PLOT].plot(xs, t_evol,
                          color=color[n], lw=2, ls=linestyle[n],
                          label=r'$\tau=' + f'{tau:.2f}$ [fm/c]'
                          if add_labels else None)
        ax_1[MU_PLOT].plot(xs, mu_evol,
                           color=color[n], lw=2, ls=linestyle[n],
                           label=second_label if n == 0 else None)

        pi_xx, pi_yy, pi_xy, pi_nn = milne_pi(
            tau,
            xs,
            xs,
            1,
            t_interp,
            mu_interp,
            pi_interp
        )

        ax_1[PIXX_PLOT].plot(xs,
                             pi_yy * HBARC ** 3 / (4.0 * e_evol / 3.0),
                             color=color[n], lw=2, ls=linestyle[n])

        # need to add code to calculate sigma^{xy}
        ax_1[PIXY_PLOT].plot(xs,
                             pi_xy * HBARC ** 3 / (4.0 * e_evol / 3.0),
                             color=color[n], lw=2, ls=linestyle[n])

        ax_2[E_PLOT].plot(xs, e_evol,  # / t_evol ** 4,
                          color=color[n], lw=2, ls=linestyle[n],
                          label=r'$\tau=' + f'{tau:.2f}$ [fm/c]'
                          if add_labels else None)
        ax_2[N_PLOT].plot(xs, n_evol,  # / t_evol ** 3,
                          color=color[n], lw=2, ls=linestyle[n])
        ax_2[S_PLOT].plot(xs, s_evol,  # / t_evol ** 3,
                          color=color[n], lw=2, ls=linestyle[n])


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(2 * 7, 2 * 7))
    fig.patch.set_facecolor('white')

    fig2, ax2 = plt.subplots(ncols=1, nrows=3, figsize=(1 * 7, 3 * 7))

    y0s = array([1.2, 1e-20, 1.5])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)
    xs = linspace(-6, 6, 200)

    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=array([1.2, 2.0, 3.0]),
        color=['black', 'black', 'black'],
        linestyle=['solid', 'dashed', 'dotted'],
        add_labels=True,
        second_label=r'$\mu_0/T_0=0.0$'
    )

    y0s = array([1.2, 2 * 1.2, 1.5])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)
    xs = linspace(-6, 6, 200)

    solve_and_plot(
        ax_1=ax,
        ax_2=ax2,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=array([1.2, 2.0, 3.0]),
        color=['red', 'red', 'red'],
        linestyle=['solid', 'dashed', 'dotted'],
        second_label=r'$\mu_0/T_0=2.0$'
    )

    costumize_axis(
        ax=ax[T_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$T(\tau, x)$ [GeV]'
    )
    costumize_axis(
        ax=ax[MU_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$\mu(\tau, x)$ [GeV]'
    )
    costumize_axis(
        ax=ax[PIXX_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$\pi^{yy}(\tau, x) / h(\tau, x)$'
    )
    costumize_axis(
        ax=ax[PIXY_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$\pi^{xy}(\tau, x) / h(\tau, x)$'
    )

    ax[T_PLOT].legend(loc='upper right', fontsize=20)
    ax[MU_PLOT].legend(loc='upper right', fontsize=20)
    fig.tight_layout()
    fig.savefig('./viscous-gubser-current-soln-1.pdf')

    costumize_axis(
        ax=ax2[E_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$\mathcal E(\tau, x)/T(\tau, x)^4$'
        # y_title=r'$\mathcal E(\tau, x)$ [GeV/fm$^{-3}$]'
    )
    costumize_axis(
        ax=ax2[N_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$n(\tau, x)/T(\tau, x)^3$'
        # y_title=r'$n(\tau, x)$ [fm$^{-3}$]'
    )
    costumize_axis(
        ax=ax2[S_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$s(\tau, x) / T(\tau, x)^3$'
    )

    ax2[E_PLOT].legend(loc='upper right', fontsize=20)
    fig2.tight_layout()
    fig2.savefig('./viscous-gubser-current-soln-2.pdf')
