# auxiliary-tools

Utility packages for atmospheric and related research workflows.

## Installation

Clone the repository:

```bash
git clone https://github.com/cstdl-atmos-snu/auxiliary-tools.git
```

## Requirements
- [Anaconda3](https://docs.anaconda.com/anaconda/install/) or [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).
- [NumPy](https://numpy.org/).
- [SciPy](https://scipy.org/).
- [Xarray](https://docs.xarray.dev/en/stable/).
- [tcpyPI](https://github.com/dgilford/tcpypi) (for `TC_related` module).
- [MetPy](https://unidata.github.io/MetPy/latest/index.html) (for `TC_related` module).

You can install the required packages with `conda` and `pip`:

```bash
conda create -n cstdl_snu -y
conda activate cstdl_snu
conda install -c conda-forge numpy scipy xarray dask netCDF4 bottleneck metpy -y
pip install tcpypi
```

## Usage
To use the custom packages, add the repository path to `sys.path` before importing:

```python
import xarray as xr
import numpy as np

import sys
sys.path.append("/path/to/auxiliary-tools")

from cstdl_tools.calc import sphere, vertical
from cstdl_tools.TC_related import environment_variables as envvar
import cstdl_tools.constants as consts
```