import os
from collections.abc import Mapping
from pathlib import Path

import yaml


class ConfigNode:
    def __init__(self, data, parent_keys=None, extra=None):
        self._data = data
        self._parent_keys = parent_keys or []
        self.extra = extra or {}

    def __getattr__(self, key):
        if key not in self._data:
            raise AttributeError(f"No such attribute: {key}")

        value = self._data[key]

        if isinstance(value, Mapping):
            return ConfigNode(value, [*self._parent_keys, key], extra=self.extra)
        typ = type(value)  # typecast according to pre-set config value
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            # The value requires an environment variable to be resolved
            env_key = value[1:-1]
            if os.environ.get(env_key):  # Ignore empty strings
                value = os.environ[env_key]
                if value.lower().strip() == "false" and typ is bool:
                    return False
                return typ(value)

        # Allow override of any key by an environment variable
        env_key = "CUTQC2_" + "_".join([*self._parent_keys, key]).upper()
        if os.environ.get(env_key):  # Ignore empty strings
            value = os.environ[env_key]
            if value.lower().strip() == "false" and typ is bool:
                return False
            return typ(value)

        if isinstance(value, str):
            for k, v in self.extra.items():
                value = value.replace(k, v)
        return value

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return repr(self._data)


class Config(ConfigNode):
    def __init__(self, yaml_file: Path):
        self.file_path = yaml_file
        self.file_dir = yaml_file.parent.resolve()

        with yaml_file.open() as f:
            data = yaml.safe_load(f)
        super().__init__(data, extra={"__dir__": str(self.file_dir)})
