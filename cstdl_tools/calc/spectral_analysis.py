import numpy as np
import xarray as xr
import scipy.fft as fft


def spectral_analysis(
    da,
    time_name="time",
    lon_name="lon",
    lat_name="lat",
    window_days=96,
    overlap_days=60,
    hann_days=5,
    num_smooth=15,
):
    """
    Assuming `da` has a regular time, lat, lon grid, and lat dimension is symmetric about the equator.
    """
    dt = da[time_name][1] - da[time_name][0]  # assume regular time axis
    dlon = da[lon_name][1] - da[lon_name][0]  # assume regular lon axis
    samples_per_day = np.timedelta64(1, "D") / dt.values
    samples_per_lon = 360.0 / dlon.values

    nseg = int(window_days * samples_per_day)
    nlon = da.sizes[lon_name]
    nlat = da.sizes[lat_name]

    step = int((window_days - overlap_days) * samples_per_day)  # 36 days
    hann_width = int(hann_days * samples_per_day)

    # symmetric and asymmetric array
    da = da.transpose(time_name, lat_name, lon_name).sortby(lat_name)
    sym_values = (da.values + da.values[:, ::-1, :]) / 2
    asym_values = (da.values - da.values[:, ::-1, :]) / 2
    ds = xr.Dataset(
        {
            "sym": ([time_name, lat_name, lon_name], sym_values[:, nlat // 2 :, :]),
            "asym": ([time_name, lat_name, lon_name], asym_values[:, nlat // 2 :, :]),
        },
        coords={
            time_name: da[time_name],
            lat_name: da[lat_name][nlat // 2 :],
            lon_name: da[lon_name],
        },
    )

    # rolling construct → shape: (..., time, window)
    rolled = ds.rolling({time_name: nseg}, center=False).construct("time_in_segment")

    # keep only fully-formed windows, then stride by `step`
    # the first (win-1) entries are NaN because the window is not full yet
    rolled = rolled.isel({time_name: slice(nseg - 1, None, step)})

    # rename the (now sparse) time axis to 'segment' for clarity
    rolled = rolled.rename({time_name: "segment"})
    rolled["segment"] = np.arange(rolled.sizes["segment"])

    # make "time_in_segment" a proper timedelta index (0..win-1 samples)
    rolled = rolled.assign_coords(
        time_in_segment=(np.arange(nseg) / samples_per_day * 24).astype(
            "timedelta64[h]"
        )
    )

    # detrend along time axis
    linear_trend = rolled["sym"].polyfit(dim="time_in_segment", deg=1)
    rolled["sym"] = rolled["sym"] - xr.polyval(
        rolled["time_in_segment"], linear_trend["polyfit_coefficients"]
    )  # linear trend
    linear_trend = rolled["asym"].polyfit(dim="time_in_segment", deg=1)
    rolled["asym"] = rolled["asym"] - xr.polyval(
        rolled["time_in_segment"], linear_trend["polyfit_coefficients"]
    )  # linear trend

    # apply Hann window along "time_in_segment" axis
    hann_window = np.concatenate(
        (
            np.hanning(hann_width)[: hann_width // 2],
            np.ones(rolled.sizes["time_in_segment"] - hann_width),
            np.hanning(hann_width)[hann_width // 2 :],
        ),
        axis=0,
    )
    hann_window = xr.DataArray(
        hann_window,
        coords={"time_in_segment": rolled["time_in_segment"]},
        dims=["time_in_segment"],
    )
    windowed = rolled * hann_window
    windowed = windowed.transpose("segment", lat_name, "time_in_segment", lon_name)

    # fourier transform along time_in_segment and lon axes
    freq = fft.fftshift(fft.fftfreq(nseg, d=1 / samples_per_day))
    wavenum = -fft.fftshift(fft.fftfreq(nlon, d=1 / samples_per_lon))
    fft2d_sym = np.fft.fft2(windowed["sym"])
    fft2d_asym = np.fft.fft2(windowed["asym"])
    power_sym_mean = np.fft.fftshift(
        np.mean(np.abs(fft2d_sym) ** 2 / (nseg * nlon), axis=(0, 1))
    )
    power_asym_mean = np.fft.fftshift(
        np.mean(np.abs(fft2d_asym) ** 2 / (nseg * nlon), axis=(0, 1))
    )

    ds = xr.Dataset(
        {
            "power_sym_mean": (["freq", "wavenum"], power_sym_mean),
            "power_asym_mean": (["freq", "wavenum"], power_asym_mean),
        },
        coords={
            "freq": freq,
            "wavenum": wavenum,
        },
    )

    # smooth the power spectrum with a [1, 2, 1] filter along both axes
    weight = xr.DataArray([0.25, 0.5, 0.25], dims=["window"])
    ds["power_sym_mean"] = (
        ds["power_sym_mean"]
        .rolling(freq=3, center=True)
        .construct(freq="window")
        .dot(weight)
        .rolling(wavenum=3, center=True)
        .construct(wavenum="window")
        .dot(weight)
    )
    ds["power_asym_mean"] = (
        ds["power_asym_mean"]
        .rolling(freq=3, center=True)
        .construct(freq="window")
        .dot(weight)
        .rolling(wavenum=3, center=True)
        .construct(wavenum="window")
        .dot(weight)
    )
    power_smooth = (ds["power_sym_mean"].copy() + ds["power_asym_mean"].copy()) / 2
    for i in range(num_smooth):
        power_smooth = (
            power_smooth.rolling(freq=3, center=True)
            .construct(freq="window")
            .dot(weight)
        )
    for i in range(num_smooth):
        power_smooth = (
            power_smooth.rolling(wavenum=3, center=True)
            .construct(wavenum="window")
            .dot(weight)
        )
    ds["power_smooth"] = power_smooth
    ds["power_sym_ratio"] = ds["power_sym_mean"] / power_smooth
    ds["power_asym_ratio"] = ds["power_asym_mean"] / power_smooth

    return ds


def spectral_filtering(
    da,
    time_name="time",
    lon_name="lon",
    lat_name="lat",
    hann_days=5,
    wave_type="MJO",
    wavenumber_band=(0, 9),
    frequency_band=(1 / 96, 1 / 30),
    equivalent_depth_band=(8, 90),
):
    """
    Assuming `da` has a regular time, lat, lon grid, and lat dimension is symmetric about the equator.
    wave_type: "MJO", "Kelvin", "ER", "MRG", "TD-type"
    """
    dt = da[time_name][1] - da[time_name][0]  # assume regular time axis
    dlon = da[lon_name][1] - da[lon_name][0]  # assume regular lon axis
    samples_per_day = np.timedelta64(1, "D") / dt.values
    samples_per_lon = 360.0 / dlon.values

    nt = da.sizes[time_name]
    nlon = da.sizes[lon_name]
    nlat = da.sizes[lat_name]

    hann_width = int(hann_days * samples_per_day)

    # symmetric and asymmetric array
    da = da.transpose(time_name, lat_name, lon_name).sortby(lat_name)

    # detrend along time axis
    linear_trend = da.polyfit(dim=time_name, deg=1)
    da = da - xr.polyval(
        da[time_name], linear_trend["polyfit_coefficients"]
    )  # linear trend

    # apply Hann window along time_name axis
    hann_window = np.concatenate(
        (
            np.hanning(hann_width)[: hann_width // 2],
            np.ones(da.sizes[time_name] - hann_width),
            np.hanning(hann_width)[hann_width // 2 :],
        ),
        axis=0,
    )
    hann_window = xr.DataArray(
        hann_window,
        coords={time_name: da[time_name]},
        dims=[time_name],
    )
    windowed = da * hann_window
    windowed = windowed.transpose(lat_name, time_name, lon_name)

    # fourier transform along time_name and lon axes
    freq = fft.fftfreq(nt, d=1 / samples_per_day)
    wavenum = -fft.fftfreq(nlon, d=1 / samples_per_lon)
    fft2d_windowed = xr.DataArray(
        np.fft.fft2(windowed),
        coords={
            lat_name: windowed[lat_name],
            "freq": freq,
            "wavenum": wavenum,
        },
        dims=[lat_name, "freq", "wavenum"],
    )

    # filtering
    beta = 2.3e-11
    Re = 6.371e6  # Earth's radius in meters
    g = 9.81  # gravitational acceleration in m/s^2

    c0 = np.sqrt(g * equivalent_depth_band[0])
    c1 = np.sqrt(g * equivalent_depth_band[1])
    k_to_star_0 = np.sqrt(c0 / beta) / Re
    k_to_star_1 = np.sqrt(c1 / beta) / Re
    w_to_star_0 = 1 / np.sqrt(c0 * beta)
    w_to_star_1 = 1 / np.sqrt(c1 * beta)
    seconds_per_cycle = 86400 / (2 * np.pi)

    if wave_type == "MJO":
        filtered = fft2d_windowed.where(
            (fft2d_windowed["wavenum"] >= wavenumber_band[0])
            & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
            & (fft2d_windowed["freq"] >= frequency_band[0])
            & (fft2d_windowed["freq"] <= frequency_band[1]),
            0,
        )
    elif wave_type == "Kelvin":
        depth_lower_bound = (
            fft2d_windowed["wavenum"] * k_to_star_0 / w_to_star_0 * seconds_per_cycle
        )
        depth_upper_bound = (
            fft2d_windowed["wavenum"] * k_to_star_1 / w_to_star_1 * seconds_per_cycle
        )
        filtered = fft2d_windowed.where(
            (fft2d_windowed["wavenum"] >= wavenumber_band[0])
            & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
            & (fft2d_windowed["freq"] >= frequency_band[0])
            & (fft2d_windowed["freq"] <= frequency_band[1])
            & (fft2d_windowed["freq"] >= depth_lower_bound)
            & (fft2d_windowed["freq"] <= depth_upper_bound),
            0,
        )
    elif wave_type == "ER":
        n = 1
        depth_lower_bound = (
            (
                -(fft2d_windowed["wavenum"] * k_to_star_0)
                / ((fft2d_windowed["wavenum"] * k_to_star_0) ** 2 + (2 * n + 1))
            )
            / w_to_star_0
            * seconds_per_cycle
        )
        depth_upper_bound = (
            (
                -(fft2d_windowed["wavenum"] * k_to_star_1)
                / ((fft2d_windowed["wavenum"] * k_to_star_1) ** 2 + (2 * n + 1))
            )
            / w_to_star_1
            * seconds_per_cycle
        )
        filtered = fft2d_windowed.where(
            (fft2d_windowed["wavenum"] >= wavenumber_band[0])
            & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
            & (fft2d_windowed["freq"] >= frequency_band[0])
            & (fft2d_windowed["freq"] <= frequency_band[1])
            & (fft2d_windowed["freq"] >= depth_lower_bound)
            & (fft2d_windowed["freq"] <= depth_upper_bound),
            0,
        )
    elif wave_type == "MRG":
        depth_lower_bound = (
            (
                fft2d_windowed["wavenum"] * k_to_star_0 / 2
                + np.sqrt((fft2d_windowed["wavenum"] * k_to_star_0 / 2) ** 2 + 1)
            )
            / w_to_star_0
            * seconds_per_cycle
        )
        depth_upper_bound = (
            (
                fft2d_windowed["wavenum"] * k_to_star_1 / 2
                + np.sqrt((fft2d_windowed["wavenum"] * k_to_star_1 / 2) ** 2 + 1)
            )
            / w_to_star_1
            * seconds_per_cycle
        )
        filtered = fft2d_windowed.where(
            (fft2d_windowed["wavenum"] >= wavenumber_band[0])
            & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
            & (fft2d_windowed["freq"] >= frequency_band[0])
            & (fft2d_windowed["freq"] <= frequency_band[1])
            & (fft2d_windowed["freq"] >= depth_lower_bound)
            & (fft2d_windowed["freq"] <= depth_upper_bound),
            0,
        )
    elif wave_type == "TD-type":
        filtered = fft2d_windowed.where(
            (fft2d_windowed["wavenum"] >= wavenumber_band[0])
            & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
            & (fft2d_windowed["freq"] >= frequency_band[0])
            & (fft2d_windowed["freq"] <= frequency_band[1]),
            0,
        )

    # inverse fourier transform
    ifft2d = xr.DataArray(
        np.fft.ifft2(filtered.values).real,
        coords={
            lat_name: windowed[lat_name],
            time_name: windowed[time_name],
            lon_name: windowed[lon_name],
        },
        dims=[lat_name, time_name, lon_name],
    ).transpose(time_name, lat_name, lon_name)

    return ifft2d
