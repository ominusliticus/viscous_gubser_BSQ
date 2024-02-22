from pathlib import Path
from typing import Optional
from typing import List
from variable_conversions import HBARC
from variable_conversions import milne_pi
from variable_conversions import milne_number
from variable_conversions import milne_energy
from variable_conversions import u_y
from variable_conversions import u_x
from equations_of_motion import eom
from numpy import concatenate
from numpy import array
from numpy import ndarray
from numpy import arange
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


if __name__ == "__main__":
    cfg = Config()

    # Config file gives things in the units indicated by the comments.
    # Here we have to convert to the corresponding dimensionless variables
    #   for de Sitter space
    temp_0 = cfg.temp_0 * cfg.tau_0 / HBARC
    muB_0 = cfg.muB_0 * cfg.tau_0 / HBARC
    muS_0 = cfg.muS_0 * cfg.tau_0 / HBARC
    muQ_0 = cfg.muQ_0 * cfg.tau_0 / HBARC
    y0s = array([temp_0, muB_0, muS_0, muQ_0, cfg.pi_0])

    ceos_temp_0 = cfg.ceos_temp_0
    ceos_mu_0 = array([cfg.ceos_muB_0, cfg.ceos_muS_0, cfg.ceos_muQ_0])
    consts = {'temperature_0': ceos_temp_0, 'chem_potential_0': ceos_mu_0}

    rhos_1 = linspace(-10, 0, 1000)[::-1]
    rhos_2 = linspace(0, 10, 1000)

    soln_1 = odeint(eom, y0s, rhos_1, args=(ceos_temp_0, ceos_mu_0,))
    soln_2 = odeint(eom, y0s, rhos_2, args=(ceos_temp_0, ceos_mu_0,))
    t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))
    mu_hat = [concatenate((soln_1[:, i][::-1], soln_2[:, i]))
              for i in [1, 2, 3]]
    pi_bar_hat = concatenate((soln_1[:, 4][::-1], soln_2[:, 4]))
    rhos = concatenate((rhos_1[::-1], rhos_2))

    t_interp = interp1d(rhos, t_hat)
    mu_interp = [interp1d(rhos, f) for f in mu_hat]
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
    dir_name = f'tau0={cfg.tau_0:.2f}_T0={cfg.temp_0:.2f}_muB0={cfg.muB_0:.2f}__muS0={cfg.muS_0:.2f}_muQ0={cfg.muQ_0:.2f}pi0={cfg.pi_0:.2f}'
    dir_path = Path(dir_name).absolute()

    try:
        os.mkdir(dir_path)
    except (FileExistsError):
        pass

    print(cfg.tau_0, cfg.tau_f, cfg.tau_step)
    for tau in linspace(cfg.tau_0, cfg.tau_f, int(
            (cfg.tau_f - cfg.tau_0) / cfg.tau_step) + 1):
        file_name = f'{dir_name}/tau={tau:.2f}.txt'
        path = Path(cfg.output_dir).absolute() / file_name
        with open(str(path), 'w') as f:
            f.write(f'#0 {stepx} {stepy} {stepEta} 0 {xmin} {ymin} {etamin}\n')

            for x in arange(xmin, xmax, stepx):
                for y in arange(ymin, ymax, stepy):
                    pis = milne_pi(
                        tau=tau,
                        x=x,
                        y=y,
                        q=1.0,
                        ads_T=t_interp,
                        ads_mu=mu_interp,
                        ads_pi_bar_hat=pi_interp,
                        **consts,
                        tol=cfg.tol
                    )
                    f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'
                            .format(
                                x,
                                y,
                                0,  # eta
                                milne_energy(
                                    tau=tau,
                                    x=x,
                                    y=y,
                                    q=1.0,
                                    ads_T=t_interp,
                                    ads_mu=mu_interp,
                                    **consts,
                                    tol=cfg.tol
                                ),
                                *milne_number(
                                    tau=tau,
                                    x=x,
                                    y=y,
                                    q=1.0,
                                    ads_T=t_interp,
                                    ads_mu=mu_interp,
                                    **consts,
                                    tol=cfg.tol
                                ),
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
