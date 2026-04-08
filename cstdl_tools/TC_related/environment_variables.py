import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import metpy.units as mpunits

from tcpyPI import pi as tcpi

from .. import constants
from ..calc import sphere
from ..calc import vertical

# physical parameters
Lv = constants.Lv
g = constants.g
c_p = constants.Cp_d
Rd = constants.Rd
Rv = constants.Rv
epsilon = constants.epsilon
T0 = 273.15


def relative_vorticity(u, v, doLonPad=True):
    """
    Calculate the relative vorticity (zeta) from the u and v wind components.

    Parameters:
        u (xarray.DataArray): The zonal wind component (in m/s).
        v (xarray.DataArray): The meridional wind component (in m/s).
    """
    zeta = sphere.dx_central(v, doLonPad=doLonPad) - sphere.dy_central(u)
    return zeta


def absolute_vorticity(zeta):
    """
    Calculate the absolute vorticity (eta) from the u and v wind components.

    Parameters:
        zeta (xarray.DataArray): The relative vorticity (in 1/s).
    """
    f = sphere.f(zeta["latitude"])
    eta = zeta + f
    return eta


def column_integrated_moisture(q):
    """
    Calculate the column-integrated moisture (W) given specific humidity.
    Parameters:
        q (xarray.DataArray): Specific humidity (in kg/kg).
    """
    W = vertical.column_integral(q)
    return W


def saturated_column_integrated_moisture(t):
    """
    Calculate the saturated column-integrated moisture (W) given temperature.
    Parameters:
        t (xarray.DataArray): Temperature (in K).
    """
    rs = xr.apply_ufunc(
        mpcalc.saturation_mixing_ratio,
        kwargs={
            "total_press": t["level"] * 100 * mpunits.units("Pa"),
            "temperature": t * mpunits.units("K"),
            "phase": "liquid",
        },
    )
    qs = xr.apply_ufunc(
        mpcalc.specific_humidity_from_mixing_ratio,
        rs,
    )
    qs.values = qs.values
    Ws = vertical.column_integral(qs)
    return Ws


def column_relative_humidity(W, Ws):
    """
    Calculate the column-relative humidity (CRH) given specific humidity and temperature.
    Parameters:
        W (xarray.DataArray): Column-integrated moisture (in kg/m^2).
        Ws (xarray.DataArray): Saturated column-integrated moisture (in kg/m^2).
    """
    CRH = W / Ws
    return CRH


def column_saturation_deficit(W, Ws):
    """
    Calculate the column saturation deficit (SD) given specific humidity and temperature.
    Parameters:
        W (xarray.DataArray): Column-integrated moisture (in kg/m^2).
        Ws (xarray.DataArray): Saturated column-integrated moisture (in kg/m^2).
    """
    SD = W - Ws
    return SD


def vertical_wind_shear(u_upper, v_upper, u_lower, v_lower):
    """
    Calculate the vertical wind shear (VWS) given zonal and meridional wind components.
    Parameters:
        u_upper (xarray.DataArray): Upper level zonal wind component (in m/s).
        v_upper (xarray.DataArray): Upper level meridional wind component (in m/s).
        u_lower (xarray.DataArray): Lower level zonal wind component (in m/s).
        v_lower (xarray.DataArray): Lower level meridional wind component (in m/s).
    """
    du = u_upper - u_lower
    dv = v_upper - v_lower
    VWS = np.sqrt(du**2 + dv**2)
    return VWS


def moist_entropy(T, p, RH):
    """
    calculate moist entropy given air temperature (T), pressure (p) and relative humidity (RH).
    The equation is: s = c_p*log(T) - Rd*log(p_d) + Lv*r_v/T - Rv*r_v*log(RH)
    parameters:
        T (xarray.DataArray): air temperature (K)
        p (xarray.DataArray): air pressure (Pa)
        RH (xarray.DataArray or scalar): relative humidity (0-1)
    """
    r_v = xr.apply_ufunc(
        mpcalc.mixing_ratio_from_relative_humidity,
        p * mpunits.units("Pa"),
        T * mpunits.units("K"),
        RH * mpunits.units("dimensionless"),
    )
    e = xr.apply_ufunc(
        mpcalc.vapor_pressure,
        p * mpunits.units("Pa"),
        r_v,
    )
    r_v.values = r_v.values  # remove units
    e.values = e.values  # remove units
    p_d = p - e
    p_d = p_d.where(p_d > 0)
    s = c_p * np.log(T) - Rd * np.log(p_d) + Lv * r_v / T - Rv * r_v * np.log(RH)
    return s


def entropy_deficit(SST, MSL, Tb, RHb, Tm, pm, RHm):
    """
    calculate entropy deficity defined in Tang and Emanuel, 2012.
    chi = (s_m_star - s_m) / (s_sst_star - s_b)
    parameters:
        SST (xarray.DataArray): sea surface temperature (in Kelvin)
        MSL (xarray.DataArray): mean sea level pressure (in Pa)
        Tb (xarray.DataArray): boundary layer air temperature (in Kelvin)
        RHb (xarray.DataArray): boundary layer relative humidity (0-1)
        Tm (xarray.DataArray): middle troposphere air temperature (in Kelvin)
        pm (xarray.DataArray): middle troposphere pressure level (in Pa)
        RHm (xarray.DataArray): middle troposphere relative humidity (0-1)
    """
    s_sst_star = moist_entropy(T=SST, p=MSL, RH=1)
    s_b = moist_entropy(T=Tb, p=MSL, RH=RHb)
    s_m_star = moist_entropy(T=Tm, p=pm, RH=1)
    s_m = moist_entropy(T=Tm, p=pm, RH=RHm)
    denominator = s_sst_star - s_b
    denominator = denominator.where(denominator > 0)
    numerator = s_m_star - s_m
    chi = numerator / denominator
    return chi


def potential_intensity(SSTC, MSL, P, TC, R):
    """
    calculate potential intensity using tcpyPI (Gilford, 2020)
    parameters:
        SSTC (xarray.DataArray): sea surface temperature (in Celsius)
        MSL (xarray.DataArray): mean sea level pressure (in hPa)
        P (xarray.DataArray): pressure levels (in hPa)
        TC (xarray.DataArray): air temperature profile (in Celsius)
        R (xarray.DataArray): mixing ratio profile (in g/kg)
    """
    vmax, pmin, ifl, t0, otl = xr.apply_ufunc(
        tcpi,
        SSTC,  # (C)
        MSL,  # (hPa)
        P,  # (hPa)
        TC,  # (C)
        R,  # (g/kg)
        input_core_dims=[[], [], ["level"], ["level"], ["level"]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
    )
    return vmax


def ventilation_index(VWS, chi, PI):
    """
    calculate ventilation index defined in Tang and Emanuel, 2012.
    VI = VWS * chi / PI
    parameters:
        VWS (xarray.DataArray): vertical wind shear (in m/s)
        chi (xarray.DataArray): entropy deficit (dimensionless)
        PI (xarray.DataArray): potential intensity (in m/s)
    """
    VI = VWS * chi / PI
    return VI


def ventilated_potential_intensity(VI, PI):
    """
    calculate ventilated potential intensity defined in Chavas et al. (2025).
    parameters:
        VI (xarray.DataArray): ventilation index (dimensionless)
        PI (xarray.DataArray): potential intensity (in m/s)
    """
    VI_max = 0.145
    VI_comp = VI + 0j
    x = (
        1
        / np.sqrt(3)
        * (np.sqrt((VI_comp / VI_max) ** 2 - 1) - VI_comp / VI_max) ** (1 / 3)
    )
    vPI_comp = (x + 1 / (3 * x)) * PI
    vPI = vPI_comp.real.where(VI < VI_max, 0)
    return vPI
