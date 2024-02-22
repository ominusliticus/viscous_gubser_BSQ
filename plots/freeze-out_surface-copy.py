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

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from my_plotting import costumize_axis

from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import denergy_drho

from variable_conversions import rho
from variable_conversions import HBARC

from typing import List


def find_freezeout_tau(
        e_interp: interp1d,
        e_freezeout: float,
        tau: float,
        q: float,
) -> float:
    value = newton(
        lambda r: e_freezeout - e_interp(rho(tau, r, q)) / tau ** 4,
        0.5
    )

    print(tau, value)
    return value


def denergy_dtau(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
) -> float:
    temperature, chem_potenial, _ = ys
    derivative = q + (1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau ** 2)
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return_value = derivative * denergy_drho(ys, rho(tau, r, q)) * tau
    return_value -= energy(temperature=temperature,
                           chem_potential=chem_potenial)
    return return_value / tau ** 5


def denergy_dr(
        ys: ndarray,
        tau: float,
        r: float,
        q: float,
) -> float:
    derivative = - q * r / tau
    derivative /= sqrt(
        1 + ((1 + (q * r) ** 2 - (q * tau) ** 2) / (2 * q * tau)) ** 2
    )
    return derivative * denergy_drho(ys, rho(tau, r, q)) / tau ** 4


def solve_and_plot(
        ax: plt.Axes,
        y0s: ndarray,
        rhos_1: ndarray,
        rhos_2: ndarray,
        taus: ndarray,
        e_freezeout: float,
        q: float,
        scale: float,
) -> None:
    soln_1 = odeint(eom, y0s, rhos_1)
    soln_2 = odeint(eom, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_hat)

    e_interp = interp1d(rhos, energy(t_hat, mu_hat))

    freezeout_times = zeros((taus.size, 2))
    normal_vectors = zeros_like(freezeout_times)
    arrows = zeros((taus.size,), dtype=FancyArrow)

    for i, tau in enumerate(taus):
        freezeout_times[i] = [
            find_freezeout_tau(
                e_interp, e_freezeout, tau, q
            ),
            tau,
        ]

    ax.scatter(
        freezeout_times[:, 0],
        freezeout_times[:, 1],
        color='red',
        s=1.0
    )

    for i, var in enumerate(freezeout_times[::10]):
        x, tau_FO = var
        _rho = rho(tau_FO, x, q)
        normal_vectors[i] = [
            - scale * denergy_dtau(
                ys=array([
                    t_interp(_rho),
                    mu_interp(_rho),
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q
            ),
            - scale * denergy_dr(
                ys=array([
                    t_interp(_rho),
                    mu_interp(_rho),
                    pi_interp(_rho)
                ]),
                tau=tau_FO,
                r=x,
                q=q
            ),
        ]

        arrows[i] = ax.arrow(
            x=x,
            y=tau_FO,
            dx=normal_vectors[i, 0],
            dy=normal_vectors[i, 1]
        )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')

    y0s = array([1.2, 1.2, 0])
    rhos_1 = linspace(-100, 0, 10_000)[::-1]
    rhos_2 = linspace(0, 100, 10_000)
    taus = linspace(1, 2, 1000)

    solve_and_plot(
        ax=ax,
        y0s=y0s,
        rhos_1=rhos_1,
        rhos_2=rhos_2,
        taus=taus,
        e_freezeout=1.0 / HBARC,
        q=1,
        scale=0.1,
    )

    costumize_axis(
        ax=ax,
        x_title=r'$r$ [fm]',
        y_title=r'$\tau_\mathrm{FO}$ [fm/c]',
    )
    fig.tight_layout()
    fig.savefig('./freeze-out-surface-copy.pdf')
