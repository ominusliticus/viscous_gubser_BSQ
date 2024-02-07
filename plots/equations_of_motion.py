from numpy import ndarray
from numpy import pi
from numpy import array
from numpy import sum as npsum
from numpy import tanh

from typing import Union

# Note on the the units: everything is expected to be in units fm

ALPHA = 3 * pi ** 2 * (16 + 105 / 4) / 90
CTAUR = 5
ETA_S = 0.2
AA = 1.0 / 3.0

# the equations of state stuff
def pressure(
        temperature: float,
        chem_potential: ndarray,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    return_value = (temperature / temperature_0) ** 2
    return_value += npsum((chem_potential / chem_potential_0) ** 2)
    return_value = AA * temperature_0 ** 4 * return_value ** 2
    return ALPHA * return_value


def energy(
        temperature: float,
        chem_potential: ndarray,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    return_value = (temperature / temperature_0) ** 2
    return_value += npsum((chem_potential / chem_potential_0) ** 2)
    return_value = temperature_0 ** 4 * return_value ** 2
    return ALPHA * return_value


def entropy(
        temperature: float,
        chem_potential: ndarray,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    temp_value = (temperature / temperature_0) ** 2
    temp_value += npsum((chem_potential / chem_potential_0) ** 2)
    return_value = 4 * AA * temperature * temp_value \
          * temperature_0 ** 2
    return ALPHA * return_value


def number(
        temperature: float,
        chem_potential: ndarray,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> ndarray:
    # Note that we assume the order is [B, S, Q]
    temp_value = (temperature / temperature_0) ** 2
    temp_value += npsum((chem_potential / chem_potential_0) ** 2)
    return_value = 4 * AA * temperature_0 ** 4 * chem_potential
    return_value *= temp_value / chem_potential_0 ** 2
    return ALPHA * return_value


def tau_R(
        temperature: float,
        chem_potential: ndarray,
        temperature_0: float,
        chem_potential_0: ndarray,
) -> float:
    e = energy(temperature=temperature, chem_potential=chem_potential,
               temperature_0=temperature_0, chem_potential_0=chem_potential_0)
    p = pressure(temperature=temperature, chem_potential=chem_potential,
               temperature_0=temperature_0, chem_potential_0=chem_potential_0)
    s = entropy(temperature=temperature, chem_potential=chem_potential,
               temperature_0=temperature_0, chem_potential_0=chem_potential_0)
    return CTAUR * ETA_S * s / (e + p)


# The equations of motion
def dT_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
        temperature_0: float,
        chem_potential_0: ndarray,
) -> Union[float, ndarray]:    
    temperature, mu_B, mu_Q, mu_S, pi_hat = ys
    chem_potential = array([mu_B, mu_S, mu_Q])
    return_value = npsum(pi_hat * (chem_potential * temperature_0 
                    / (chem_potential_0 * temperature)) ** 2)
    return_value += (1 / 3) * (-2 + pi_hat)
    return temperature * return_value * tanh(rho)


def dmu_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
        temperature_0: float,
        chem_potential_0: ndarray,
) -> Union[float, ndarray]:
    _, mu_B, mu_Q, mu_S, pi_hat = ys
    chem_potential = array([mu_B, mu_S, mu_Q])
    return_value = 1 + pi_hat
    return -(2 / 3) * chem_potential * return_value * tanh(rho)


def dpi_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
        temperature_0: float,
        chem_potential_0: ndarray,
) -> Union[float, ndarray]:
    temperature, mu_B, mu_Q, mu_S, pi_hat = ys
    chem_potential = array([mu_B, mu_S, mu_Q])
    tau_r = tau_R(temperature, chem_potential, temperature_0, chem_potential_0)
    return_value = (4 / 3 / CTAUR) * tanh(rho)
    return_value -= pi_hat / tau_r
    return_value -= (4 / 3) * pi_hat ** 2 * tanh(rho)
    return return_value
    

def eom(
        ys: ndarray,
        rho: Union[float, ndarray],
        temperature_0: float,
        chem_potential_0: ndarray,
) -> ndarray:
    dTdrho = dT_drho(ys, rho, temperature_0, chem_potential_0)
    dmudrho = dmu_drho(ys, rho, temperature_0, chem_potential_0)
    dpidrho = dpi_drho(ys, rho, temperature_0, chem_potential_0)

    return array([dTdrho, *dmudrho.reshape(-1,), dpidrho])


def denergy_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
        temperature_0: float,
        chem_potential_0: ndarray,
) -> ndarray:
    temperature, mu_B, mu_Q, mu_S, pi_hat = ys
    chem_potential = array([mu_B, mu_S, mu_Q])
    temp_value = (temperature / temperature_0) ** 2
    temp_value += npsum((chem_potential / chem_potential_0) ** 2)
    return_value = (
        temperature * dT_drho(ys, rho, temperature_0, chem_potential_0) 
        / temperature_0 ** 2
        +
        npsum(chem_potential * dmu_drho(ys, rho, temperature_0, chem_potential_0)
        / chem_potential_0 ** 2)
    )
    return 12 * ALPHA * temperature_0 ** 4 * temp_value * return_value