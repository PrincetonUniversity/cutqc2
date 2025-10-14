from pathlib import Path

from cutqc2.data.ondemand import get_file, list_files


def test_ondemand_file(tmp_path):
    first_filename = list_files()[0]
    filepaths = get_file(first_filename, path=tmp_path)

    # We get a list of filepaths (strings) back from the `get_file` function.
    assert len(filepaths) > 0
    assert Path(filepaths[0]).exists()
