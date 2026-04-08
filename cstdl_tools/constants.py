# from https://github.com/Unidata/MetPy/blob/v1.6.3/src/metpy/constants/default.py
# original license:
# Copyright (c) 2008,2015,2016,2018,2021 MetPy Developers.
# modified by Muchan Kim 2025-03-31
# all constants are in SI units except for pressure (hPa)

# Earth
earth_gravity = g = 9.80665  # m / s^2
Re = earth_avg_radius = 6371008.7714  # m
G = gravitational_constant = 6.67430e-11  # m^3 / (kg * s^2)
GM = geocentric_gravitational_constant = 3986005e8  # m^3 / s^2
omega = earth_avg_angular_vel = 7292115e-11  # rad / s
d = earth_sfc_avg_dist_sun = 149597870700.0  # m
S = earth_solar_irradiance = 1360.8  # W / m^2
delta = earth_max_declination = 23.45  # deg
earth_orbit_eccentricity = 0.0167  # dimensionless
earth_mass = me = geocentric_gravitational_constant / gravitational_constant  # kg

# molar gas constant
R = 8.314462618  # J / (mol * K)

# Water
Mw = water_molecular_weight = 18.015268  # kg / mol
Rv = water_gas_constant = R / Mw  # J / (kg * K)
rho_l = density_water = 999.97495  # kg / m^3 by VSMOW (1atm, 3.984 C)
wv_specific_heat_ratio = 1.330  # dimensionless (20 C)
Cp_v = wv_specific_heat_press = (
    wv_specific_heat_ratio * Rv / (wv_specific_heat_ratio - 1)  # J / (kg * K)
)
Cv_v = wv_specific_heat_vol = Cp_v / wv_specific_heat_ratio  # J / (kg * K)
Cp_l = water_specific_heat = 4.2194  # J / (kg * K)
Lv = water_heat_vaporization = 2.50084e6  # J / kg
Lf = water_heat_fusion = 3.337e5  # J / kg
Cp_i = ice_specific_heat = 2090  # J / (kg * K)
rho_i = density_ice = 917  # kg / m^3
sat_pressure_0c = 6.112  # hPa

# Dry air
Md = dry_air_molecular_weight = 28.96546e-3  # kg / mol
Rd = dry_air_gas_constant = R / Md  # J / (kg * K)
dry_air_spec_heat_ratio = 1.4  # dimensionless
Cp_d = dry_air_spec_heat_press = (
    dry_air_spec_heat_ratio * Rd / (dry_air_spec_heat_ratio - 1)  # J / (kg * K)
)
Cv_d = dry_air_spec_heat_vol = Cp_d / dry_air_spec_heat_ratio  # J / (kg * K)
rho_d = dry_air_density_stp = 1000.0 / (Rd * 273.15)  # kg / m^3 (1000 hPa, 0 C)

# General meteorology constants
P0 = pot_temp_ref_press = 1000.0  # hPa
kappa = poisson_exponent = Rd / Cp_d  # dimensionless
gamma_d = dry_adiabatic_lapse_rate = g / Cp_d  # K / m
epsilon = molecular_weight_ratio = Mw / Md  # dimensionless
