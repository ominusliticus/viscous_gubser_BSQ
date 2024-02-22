from scipy.integrate import odeint
from scipy.interpolate import interp1d

from numpy import linspace
from numpy import ndarray
from numpy import array
from numpy import concatenate
from numpy import power
from numpy import tanh

import matplotlib.pyplot as plt

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import tau_R
from equations_of_motion import ETA_S
from equations_of_motion import energy
from equations_of_motion import pressure
from equations_of_motion import entropy

from variable_conversions import rho

from typing import List
from typing import Tuple


def get_navier_stokes_ic(
        temp: float, mu: float, eta_s: float, tau: float, x: float
) -> Tuple[float, float]:
    m_p = pressure(temp, mu)
    m_s = entropy(temp, mu)

    eta = eta_s * m_s
    rhos = rho(tau, x, 1)

    # calculate P_L and P_T for navier-stokes initial conditions
    pt = m_p + ((2 / 3) * eta) * tanh(rhos)
    pl = m_p - ((4 / 3) * eta) * tanh(rhos)

    # return (2 / 3) * (pl - pt)
    return (4 / 3) * eta * tanh(rhos)


def solve_and_plot(
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        xs: ndarray,
        taus: ndarray,
        color: List[str],
) -> None:
    soln_1 = odeint(eom, y0s, rhos_1)
    soln_2 = odeint(eom, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    tau_r_hat = tau_R(temperature=t_hat, chem_potential=mu_hat)
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)
    tau_r_interp = interp1d(rhos, tau_r_hat)
    e_interp = interp1d(rhos, energy(t_hat, mu_hat))
    p_interp = interp1d(rhos, pressure(t_hat, mu_hat))

    alpha = 1.0
    for n, x in enumerate(xs):
        pi_hat = pi_interp(rho(taus, x, 1))
        tau_r = tau_r_interp(rho(taus, x, 1))
        ts = t_interp(rho(taus, x, 1))
        mus = mu_interp(rho(taus, x, 1))
        es = e_interp(rho(taus, x, 1))
        ps = p_interp(rho(taus, x, 1))
        pi_NS = array([
            get_navier_stokes_ic(tt, mm, ETA_S, tau, x)
            for tau, tt, mm in zip(taus, ts, mus)
        ])
        ax.plot(taus / tau_r, pi_hat, lw=2, color=color[0])
        ax.plot(taus / tau_r, pi_NS / (es + ps), color='black', ls='dashed')


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')

    itrs = 20
    cmap = plt.get_cmap('bwr', itrs ** 3)
    # for i, e_0 in enumerate(linspace(1.0, 2.0, itrs)):
    #     for j, n_0 in enumerate(linspace(1.0, 2.0, itrs)):
    #         for k, pi_0 in enumerate(linspace(0.0, 0.5, itrs)):
    #             y0s = array([e_0, n_0, pi_0])
    for pi_0, i in enumerate(linspace(0.0, 1.0, itrs)):
        y0s = array([1.2, 1.2, pi_0])
        rhos_1 = linspace(-10, 0, 1000)[::-1]
        rhos_2 = linspace(0, 10, 1000)
        xs = array([0])

        solve_and_plot(
            ax=ax,
            y0s=y0s,
            rhos_1=rhos_1,
            rhos_2=rhos_2,
            xs=xs,
            taus=power(10, linspace(-2, 3, 1000)),
            color=[cmap(i)],
        )

    costumize_axis(
        ax=ax,
        x_title=r'$\tau / \tau_R$',
        y_title=r'$\hat\pi/(\hat{ \mathcal E} + \hat{ \mathcal P})$'
    )
    ax.set_xscale('log')
    # ax.set_xlim(right=10)
    fig.tight_layout()
    fig.savefig('./viscous-gubser-current-attractor.pdf')
