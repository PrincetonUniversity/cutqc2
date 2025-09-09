## Contributing to the code

- Make sure that the package is installed with the `dev` extra as described in the [installation](installation.md) instructions.

  **Start a new branch**, with your initials as the prefix, for example, `js/my_awesome_feature`.

  ```
  cd </path/to/repo/clone>
  git checkout -b <initials>/<feature_or_bugfix_name>
  ```

- Install the [pre-commit](https://pre-commit.com/) package:
  ```
  pip install pre-commit
  ```
  
- Install the `pre-commit` hook. This will allow you to identify style/formatting/coding issues every time you commit your code. Pre-commit automatically formats the files in your repository according to certain standards, and/or warns you if certain best practices are not followed.
  ```
  pre-commit install
  ```

- _Tweak/modify the code, make the code better!_
- Write unit tests for your proposed changes. A drop in test coverage will cause the CI to fail, and your PR may not be merged.
- Send a PR towards the `main` branch.

Our CI will automatically run the `pre-commit` and `pytest` steps for PRs towards the protected branches, as well as checking test coverage, so running these steps on your local installation will prevent surprises for you later.

### Running the code without a GPU

It is currently possible to run the package without a GPU, and this functionality is tested in our CI (by running tests on the `macos-latest` runner, for example). This functionality *should be preserved* going forward,
as it is useful for developers who want to quickly run parts of the code on their laptops without a GPU.

To allow the package to be used without a GPU, pay attention to the following guidelines:

- If you are developing code on a machine that has a local GPU, do the proposed changes work with the GPU disabled? If you want to test the code without a GPU, you can temporarily set the `CUDA_VISIBLE_DEVICES` environment variable to an empty string in your terminal session:
  ```
  export CUDA_VISIBLE_DEVICES=""
  pytest
  ```

- Do not import `cupy` at the top level of any module, or use functions from `cupy` directly. Instead, import the `xp` module from `cutqc2` instead. This variable delegates calls to either `cupy` or `numpy` depending on which library is available and usable (Read on to see how to configure this).
  ```
  from cutqc2.numeric import xp
  my_array = xp.zeros(10)
  ```
  
- Test the code with the `CUTQC2_CORE_NUMERIC` environment variable set to `numpy`:
  The `CUTQC2_CORE_NUMERIC` environment variable can be set to either `cupy` (the default if `cupy` is available) or `numpy`. Test the code with both combinations if possible, or just `numpy` if you don't have a GPU.
  ```
  export CUTQC2_CORE_NUMERIC=cupy
  pytest
  
  export CUTQC2_CORE_NUMERIC=numpy
  pytest
  ```

### A note on our self-hosted runner

  In addition to Github-hosted runners, the project currently uses a [self-hosted](https://docs.github.com/en/actions/concepts/runners/self-hosted-runners) runner at the Princeton [Research Computing](https://researchcomputing.princeton.edu/) group.
  This runner is configured to run CI jobs on a host with multiple GPUs, and can be used for testing parts of the code that are single/multi-GPU enabled, with our without MPI.
  
  If the CI fails due to a reason that you cannot explain or fix, contact the maintainers, and we will try to help you out.