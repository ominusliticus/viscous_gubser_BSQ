from pathlib import Path
from typing import Optional
from typing import List
from variable_conversions import HBARC
from equations_of_motion import eom
from equations_of_motion import energy
from equations_of_motion import pressure
from equations_of_motion import number
from numpy import concatenate
from numpy import array
from numpy import linspace
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import sys
import os

sys.path.append('../')


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
                    self.tau_step == float(value)
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

    # Write header
    dir_name = f'tau0={cfg.tau_0:.2f}_T0={cfg.temp_0:.2f}_mu0={cfg.mu_0:.2f}_pi0={cfg.pi_0:.2f}'
    dir_path = Path(dir_name).absolute()

    try:
        os.mkdir(dir_path)
    except (FileExistsError):
        pass

    file_name = f'{dir_name}/init_conditions.txt'
    path = Path(file_name).absolute()
    with open(str(path), 'w') as f:
        for rh in linspace(-20, 20, 4000):
            t = t_interp(rh)
            mu = mu_interp(rh)
            e = energy(t, mu)
            p = pressure(t, mu)
            n = number(t, mu)
            f.write('{} {} {} {}\n'.format(rh, e, n, pi_interp(rh) * (e + p)))
