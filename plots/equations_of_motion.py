from numpy import cosh
from numpy import sinh
from numpy import tanh
from numpy import ndarray
from numpy import pi
from numpy import array

from typing import Union

# Note on the the units: everything is expected to be in units fm

NC = 3
NF = 2.5
ALPHA = (2 * (NC **2 - 1) + 7 * NC * NF / 2)
BETA = 4 * NC * NF
CTAUR = 5
ETA_S = 0.2

# the equations of state stuff


def pressure(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = BETA * chem_potential ** 2 * temperature ** 2 / 216
    return_value += BETA * chem_potential ** 4 / 324 / pi ** 2
    return_value += ALPHA * pi ** 2 * temperature ** 4 / 90
    return return_value


def energy(
        temperature: float,
        chem_potential: float,
) -> float:
    return 3 * pressure(temperature, chem_potential)


def entropy(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = 2 * pi ** 2 * ALPHA * temperature ** 3 / 45
    return_value += BETA * temperature * chem_potential ** 2 / 108
    return return_value


def number(
        temperature: float,
        chem_potential: float,
) -> float:
    return_value = BETA * temperature ** 2 * chem_potential / 108
    return_value += BETA * chem_potential ** 3 / 81 / pi ** 2
    return ALPHA * return_value


def tau_R(
        temperature: float,
        chem_potential: float,
) -> float:
    e = energy(temperature=temperature, chem_potential=chem_potential)
    p = pressure(temperature=temperature, chem_potential=chem_potential)
    s = entropy(temperature=temperature, chem_potential=chem_potential)
    return CTAUR * ETA_S * s / (e + p)


# The equations of motion
def f(
        temperature: float,
        chem_potential: float,
) -> float:
    numerator = 5 * BETA * 3 * pi ** 2 * chem_potential ** 2 * temperature ** 2
    numerator += 5 * BETA * 2 * chem_potential ** 4
    numerator += 36 * pi ** 2 * ALPHA * temperature ** 4

    denominator = 72 * pi ** 4 * ALPHA * temperature ** 4
    denominator += 288 * pi ** 2 * ALPHA * temperature ** 2 * chem_potential ** 2
    denominator += -15 * pi ** 2 * BETA * chem_potential ** 2 * temperature ** 2
    denominator += 20 * BETA * chem_potential ** 4
    return numerator / denominator


def dT_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:

    temperature, chem_potential, pi_hat = ys
    return_value = 1 + (4 * chem_potential ** 2) / (pi ** 2 * temperature ** 2)
    return_value *= -f(temperature, chem_potential) * pi_hat
    return_value += 1
    return -(2 / 3) * temperature * return_value * tanh(rho)


def dmu_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    temperature, chem_potential, pi_hat = ys
    return_value = 1 + 2 * f(temperature, chem_potential) * pi_hat
    return -(2 / 3) * chem_potential * return_value * tanh(rho)


def dpi_drho(
        ys: ndarray,
        rho: Union[float, ndarray]
) -> Union[float, ndarray]:
    temperature, chem_potenial, pi_hat = ys
    tau_r = tau_R(temperature=temperature, chem_potential=chem_potenial)
    return_value = (4 / 3 / CTAUR) * tanh(rho)
    return_value -= pi_hat / tau_r
    return_value -= (4 / 3) * pi_hat ** 2 * tanh(rho)
    return return_value


def eom(
        ys: ndarray,
        rho: Union[float, ndarray],
        ideal: bool = False
) -> ndarray:
    dTdrho = dT_drho(ys, rho)
    dmudrho = dmu_drho(ys, rho)
    dpidrho = 0 if ideal else dpi_drho(ys, rho)

    return array([dTdrho, dmudrho, dpidrho])


def denergy_drho(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> ndarray:
    temperature, chem_potential, _ = ys
    return_value_1 = 24 * ALPHA * pi ** 2 * temperature ** 2 
    return_value_1 += 5 * BETA * chem_potential ** 2
    return_value_1 *= 3 * temperature * dT_drho(ys, rho)
    return_value_2 = 3 * pi **2 * temperature ** 2
    return_value_2 += 4 * chem_potential ** 2
    return_value_2 *= 5 * BETA * chem_potential * dmu_drho(ys, rho) / pi ** 2
    return (return_value_1 + return_value_2) / 540


# equations of motion for alternative EoS
def dT_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:

    temperature, chem_potenial, pi_hat = ys
    return_value = (-2 + pi_hat) / 3.0
    return_value += pi_hat * (chem_potenial / pi / temperature) ** 2
    return temperature * return_value * tanh(rho)


def dmu_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> Union[float, ndarray]:
    _, chem_potenial, pi_hat = ys
    return_value = 1 + pi_hat
    return -(2 / 3) * chem_potenial * return_value * tanh(rho)


def dpi_drho_alt(
        ys: ndarray,
        rho: Union[float, ndarray]
) -> Union[float, ndarray]:
    temperature, chem_potenial, pi_hat = ys
    tau_r = tau_R(temperature=temperature, chem_potential=chem_potenial)
    return_value = (4 / 3 / CTAUR) * tanh(rho)
    return_value -= pi_hat / tau_r
    return_value -= (4 / 3) * pi_hat ** 2 * tanh(rho)
    return return_value


def eom_alt(
        ys: ndarray,
        rho: Union[float, ndarray],
) -> ndarray:
    dTdrho = dT_drho_alt(ys, rho)
    dmudrho = dmu_drho_alt(ys, rho)
    dpidrho = dpi_drho_alt(ys, rho)

    return array([dTdrho, dmudrho, dpidrho])
