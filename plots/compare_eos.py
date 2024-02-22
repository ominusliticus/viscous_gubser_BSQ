from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import concatenate

import matplotlib.pyplot as plt

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import eom_alt

from variable_conversions import milne_T
from variable_conversions import milne_mu
from variable_conversions import milne_pi

from typing import List
from typing import Callable
from typing import Union

T_PLOT = (0,)
MU_PLOT = (1,)
PIXX_PLOT = (2,)
PIXY_PLOT = (3,)


def solve_and_plot(
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        taus: ndarray,
        eq_of_motion: Callable[[ndarray, Union[float, ndarray]], ndarray],
        color: List[str],
        linestyle: List[str],
        add_labels: bool = False,
) -> None:
    soln_1 = odeint(eq_of_motion, y0s, rhos_1)
    soln_2 = odeint(eq_of_motion, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)

    for n, tau in enumerate(taus):
        ax[T_PLOT].plot(xs, milne_T(tau, xs, 1, t_interp),
                        color=color[n], lw=2, ls=linestyle[n],
                        label=r'$\tau=' + f'{tau:.2f}$ [fm/c]'
                        if add_labels else None)
        ax[MU_PLOT].plot(xs, milne_mu(tau, xs, 1, mu_interp),
                         color=color[n], lw=2, ls=linestyle[n])

        pi_xx, pi_yy, pi_xy, pi_nn = milne_pi(
            tau, xs, xs, 1, t_interp, mu_interp, pi_interp)
        ax[PIXX_PLOT].plot(xs, pi_yy, color=color[n], lw=2, ls=linestyle[n])
        ax[PIXY_PLOT].plot(xs, pi_xy, color=color[n], lw=2, ls=linestyle[n])


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(4 * 7, 1 * 7))
    fig.patch.set_facecolor('white')

    y0s = array([1.2, 2.4, 0])
    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)
    xs = linspace(-6, 6, 200)

    solve_and_plot(
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=array([1.2, 2.0, 3.0]),
        eq_of_motion=eom,
        color=['black', 'red', 'blue'],
        linestyle=['solid', 'dashed', 'dotted'],
        add_labels=True
    )

    solve_and_plot(
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        xs=xs,
        taus=array([1.2, 2.0, 3.0]),
        eq_of_motion=eom_alt,
        color=['gray', 'gray', 'gray'],
        linestyle=['dashed', 'dashed', 'dashed']
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
        y_title=r'$\pi^{yy}(\tau, x)$ [GeV/fm$^{-3}$]'
    )
    costumize_axis(
        ax=ax[PIXY_PLOT],
        x_title=r'$x$ [fm]',
        y_title=r'$\pi^{xy}(\tau, x)$ [GeV/fm$^{-3}$]'
    )

    ax[T_PLOT].legend(fontsize=20)
    fig.tight_layout()
    fig.savefig('./viscous-gubser-current-eos-comp.pdf')
