from pathlib import Path

import pooch

"""
Large files that are downloaded/cached automagically using the pooch library.

  How to add new entries to this list:

  Get expected hash by running md5sum or md5 on the file
  For Dropbox links, use "Copy Link" to get the URL,
    replace "dl=0" with "dl=1"
"""

CACHE_PATH = pooch.os_cache("cutqc2")

files = {
    # Keep the first file in here small because it's pulled in for unit-testing
    "random_16q_5s.zarr": {
        "url": "https://www.dropbox.com/scl/fi/6eww3xczkhwi5w9uvxitq/random_16q_5s.zarr.zip?rlkey=qgv0exu12e8a250g2tuv9evzj&st=prml618h&dl=1",
        "hash": "md5:cfe133d1ab8263b38bb2c45b75233061",
        "processor": pooch.Unzip(),
    },
}


def get_file(which, path: Path | None = None, return_download_path: bool = False):
    assert which in files, f"Unknown file {which}"
    file = files[which]
    download_path = path or CACHE_PATH
    downloaded_files = pooch.retrieve(
        url=file["url"],
        known_hash=file["hash"],
        fname=which,
        path=download_path,
        processor=file.get("processor"),
    )

    if return_download_path:
        return download_path
    return downloaded_files


def list_files():
    return list(files.keys())
