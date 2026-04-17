# Installation
`git clone https://github.com/cstdl-atmos-snu/auxiliary-tools.git`

# Prerequisites
- [Anaconda3](https://docs.anaconda.com/anaconda/install/) or [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).
- [NumPy](https://numpy.org/).
- [SciPy](https://scipy.org/).
- [Xarray](https://docs.xarray.dev/en/stable/).
- [tcpyPI](https://github.com/dgilford/tcpypi) (for TC_related).
- [MetPy](https://unidata.github.io/MetPy/latest/index.html) (for TC_related).

The prerequisites can readily be installed using the `conda` package manager:

```
conda create -n cstdl_snu -c conda-forge numpy scipy xarray dask netCDF4 bottleneck tcpyPI MetPy
conda activate cstdl_snu
```