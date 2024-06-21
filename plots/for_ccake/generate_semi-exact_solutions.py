import sys
import os

sys.path.append('../')

from pathlib import Path  # noqa
from typing import Optional  # noqa
from typing import List  # noqa
from variable_conversions import HBARC  # noqa
from variable_conversions import milne_pi  # noqa
from variable_conversions import milne_number  # noqa
from variable_conversions import milne_energy  # noqa
from variable_conversions import u_y  # noqa
from variable_conversions import u_x  # noqa
from equations_of_motion import eom  # noqa
from numpy import concatenate  # noqa
from numpy import array  # noqa
from numpy import ndarray  # noqa
from numpy import arange  # noqa
from numpy import linspace  # noqa
from scipy.interpolate import interp1d  # noqa
from scipy.integrate import odeint  # noqa


class Config:
    def __init__(self):
        self.tau_0: Optional[float] = None
        self.tau_f: Optional[float] = None
        self.tau_step: Optional[float] = None
        self.temp_0: Optional[float] = None
        self.mu_0: Optional[float] = None
        self.pi_0: Optional[float] = None
        self.tol: Optional[float] = None
        self.output_dir: Optional[str] = None

        self.read_from_config()

    def read_from_config(self):
        with open('run.cfg', 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split()[:2]
                if key == 'tau_0':
                    self.tau_0 = float(value)
                elif key == 'tau_f':
                    self.tau_f = float(value)
                elif key == 'tau_step':
                    self.tau_step = float(value)
                elif key == 'temp_0':
                    self.temp_0 = float(value)
                elif key == 'mu_0':
                    self.mu_0 = float(value)
                    if self.mu_0 == 0:
                        self.mu_0 = 1e-20
                elif key == 'pi_0':
                    self.pi_0 = float(value)
                elif key == 'tolerance':
                    self.tol = float(value)
                elif key == 'output_dir':
                    self.output_dir = value


if __name__ == "__main__":
    cfg = Config()

    # Config file gives things in the units indicated by the comments.
    # Here we have to convert to the corresponding dimensionless variables
    #   for de Sitter space
    temp_0 = cfg.temp_0 * cfg.tau_0 / HBARC
    mu_0 = cfg.mu_0 * cfg.tau_0 / HBARC
    y0s = array([temp_0, mu_0, cfg.pi_0])

    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)

    soln_1 = odeint(eom, y0s, rhos_1)
    soln_2 = odeint(eom, y0s, rhos_2)
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
    pi_bar_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = interp1d(rhos, mu_hat)
    pi_interp = interp1d(rhos, pi_bar_hat)

    stepx = .02
    stepy = .02
    stepEta = 0.1
    xmax = 5
    ymax = 5
    xmin = -xmax
    ymin = -ymax
    etamin = -0.1
    hbarc = 0.1973269804

    # Write header
    dir_name = f'tau0={cfg.tau_0:.2f}_T0={cfg.temp_0:.2f}_mu0={cfg.mu_0:.2f}_pi0={cfg.pi_0:.2f}'
    dir_path = Path(dir_name).absolute()

    try:
        os.mkdir(dir_path)
    except (FileExistsError):
        pass

    for tau in linspace(cfg.tau_0, cfg.tau_f, int(
            (cfg.tau_f - cfg.tau_0) / cfg.tau_step) + 1):
        file_name = f'{dir_name}/tau={tau:.2f}.txt'
        path = Path(cfg.output_dir).absolute() / file_name
        with open(str(path), 'w') as f:
            f.write(f'#0 {stepx} {stepy} {stepEta} 0 {xmin} {ymin} {etamin}\n')

            for x in arange(xmin, xmax, stepx):
                for y in arange(ymin, ymax, stepy):
                    pis = milne_pi(
                        tau,
                        x,
                        y,
                        1.0,
                        t_interp,
                        mu_interp,
                        pi_interp,
                        cfg.tol)
                    f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'
                            .format(
                                x,
                                y,
                                0,  # eta
                                milne_energy(
                                    tau, x, y, 1.0, t_interp, mu_interp, cfg.tol),
                                milne_number(
                                    tau, x, y, 1.0, t_interp, mu_interp, cfg.tol),
                                0,  # rho_S
                                0,  # rho_Q
                                u_x(tau, x, y, 1.0),
                                u_y(tau, x, y, 1.0),
                                0,  # u_eta
                                0,  # bulk
                                pis[0],  # pi^xx
                                pis[2],  # pi^xy
                                0,  # pi^xeta
                                pis[1],  # pi^yy
                                0,  # pi^yeta
                                pis[3],  # pi^etaeta
                            ))
