# Installation

Before you can use the `cutqc2` package, ensure that your system meets the following prerequisites:

- Python 3.12 or later
- Pip package manager

Clone the repository and enter it:
   ```
   git clone https://github.com/PrincetonUniversity/cutqc2.git
   cd cutqc2
   ```

## Creating a cutqc2 environment

All of `cutqc2`'s dependencies are on [PyPI](https://pypi.org/). We have developed and tested `cutqc2` on Python 3.12, but it should work on later Python versions as well.
You can create a new virtual environment using `venv`, and install dependencies using `pip`.

1. Verify that Python 3.12 or newer is installed on your system.
   ```
   python --version
   ```

2. Create a new environment and activate it.
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. In the activated environment, install the package in editable mode, along with its `dev` and `docs` extras:
    ```
    pip install -e .[dev,docs]
    ```
   
   If you're on Linux with an NVIDIA GPU and want to use GPU acceleration, you can install the `gpu` extra as well:
    ```
    pip install -e .[dev,docs,gpu]
    ```

### cutqc2 environment using conda

If you prefer using `conda`, you can use the provided `environment.yml` file to create a new conda environment with all the necessary dependencies pinned to versions that have worked for us in the past.
> **Note**
>
> `cutqc2` can get all its dependencies from `pypi` using `pip` and does not need [conda](https://docs.anaconda.com/miniconda/) for environment management.
Nevertheless, this might be the easiest option for most users who already have access to the `conda` executable locally or through a research cluster.
**It can therefore be used without getting a business or enterprise license from Anaconda. (See [Anaconda FAQs](https://www.anaconda.com/pricing/terms-of-service-faqs))**

1. Create a new conda environment named `cutqc2` with Python version 3.12.
   ```
   conda create --name cutqc2 python=3.12 pip
   ```

2. Activate the environment.
   ```
   conda activate cutqc2
   ```
   The command prompt will change to indicate the new conda environment by prepending `(cutqc2)`.


### Using a different Python version than the system default

If your Python version is not 3.12 or later, or if you're getting errors when using a non-tested Python version, we recommend using the [uv](https://github.com/astral-sh/uv) tool to create a virtual environment with the correct Python version.
`uv` is quick to [install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and easy to use, both locally as well as on research clusters.

Once `uv` is installed:

1. Create a new environment with Python 3.12 and activate it.
   ```
   uv venv --python 3.12
   source .venv/bin/activate
   ```

2. In the activated environment, install the package in editable mode, along with its `dev` and `docs` extras:
    ```
    uv pip install -e .[dev,docs]
    ```

    If you're on Linux with an NVIDIA GPU and want to use GPU acceleration, you can install the `gpu` extra as well:
   ```
    uv pip install -e .[dev,docs,gpu]
    ```

### Using "proven" dependency versions

Regardless of whether you use `uv`, python `venv`, or `conda`, if you're having trouble with installation of `cutqc2` and simply want to get it to work, 
you will want to use the exact dependency versions that we have previously tested `cutqc2` with. You can install them using the provided `requirements.txt` file.

1. If you're *not* on a Mac, in the activated environment, install the dependencies provided in `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
   
   If you're on a Mac, in the activated environment, install the dependencies provided in  `requirements-macos.txt` file instead:
    ```
    pip install -r requirements-macos.txt
    ```
   
2. In the activated environment, install the package in editable mode **without dependencies**.
    ```
    pip install -e . --no-deps
    ```

## Additional installation requirements

### Gurobi

If you want to utilize circuit-cutting in `cutqc2`, you will need to install [Gurobi](https://www.gurobi.com/solutions/licensing/) and obtain a license.
Small circuits will likely work without a license, but a valid license is needed for cutting moderate to large-sized circuits. Once you get an appropriate
license, you will likely need to set the `GRB_LICENSE_FILE` environment variable. Alternately, you can use the following environment variables, which we
use in our CI setup:

```
GUROBI_WLSACCESSID
GUROBI_LICENSEID
GUROBI_WLSSECRET
```

### MPI

Whether or not you utilize MPI support with `cutqc2` (most helpful when running the `cutqc2 postprocess` command on large datasets), you will need to have an MPI implementation installed on your system.
Most clusters will already have an MPI implementation installed, which you can load using `module load` commands. If you want to install one locally, we recommend using [OpenMPI](https://www.open-mpi.org/).
On MACs, you can install it using [Homebrew](https://brew.sh/):
```
brew install open-mpi
```

If you're able to run `mpirun` or `mpiexec` commands, you should be all set.

## Test the package

To verify that the installation was successful, you should now run the test suite using `pytest`.
   ```
   pytest
   ```

   Tests should take ~2 minutes to run. If any of the tests fail, **do not continue**, but open an issue on the [issues page](https://github.com/PrincetonUniversity/cutqc2/issues) instead, and we will try to help you out.