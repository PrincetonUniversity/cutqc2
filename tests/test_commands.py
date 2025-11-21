import subprocess
from pathlib import Path

from cutqc2 import __version__
from cutqc2.data.ondemand import list_files

THIS_FOLDER = Path(__file__).parent


def test_cmd_version():
    result = subprocess.run(
        ["cutqc2", "--version"], capture_output=True, text=True, check=False
    )
    assert __version__ in result.stdout


def test_cmd_download_list():
    result = subprocess.run(
        ["cutqc2", "download", "--list"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0


def test_cmd_download_first(tmp_path):
    file_to_download = list_files()[0]
    result = subprocess.run(
        ["cutqc2", "download", "--file", file_to_download, "--path", f"{tmp_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_script():
    # Test all commands in their logical sequence, one by one, by
    # executing a pre-packaged .sh script in the examples/scripts folder.
    cwd = Path(THIS_FOLDER / "../examples/scripts").resolve()

    result = subprocess.run(
        ["bash", "supremacy_6qubit.sh"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )

    print("\nSTDOUT\n---------\n")  # noqa: T201
    print(result.stdout)  # noqa: T201
    print("\nSTDERR\n---------\n")  # noqa: T201
    print(result.stderr)  # noqa: T201
    print("\n---------\n")  # noqa: T201

    if result.returncode != 0:
        raise RuntimeError("script execution failed")
