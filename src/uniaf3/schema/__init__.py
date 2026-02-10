"""Schemas for UniAF3 input configs."""

from pathlib import Path

from uniaf3.schema.alphafold3 import AF3Config
from uniaf3.schema.alphafold3_server import AF3ServerConfig
from uniaf3.schema.base import UniAF3BaseConfig, UniAF3Config
from uniaf3.schema.boltz import BoltzConfig
from uniaf3.schema.chai import ChaiConfig
from uniaf3.schema.protenix import ProtenixConfig

__all__ = [
    "UniAF3BaseConfig",
    "UniAF3Config",
    "AF3Config",
    "AF3ServerConfig",
    "BoltzConfig",
    "ChaiConfig",
    "ProtenixConfig",
    "dump_config",
    "write_config",
    "AnyConfig",
]

AnyConfig = (
    UniAF3Config
    | AF3Config
    | AF3ServerConfig
    | BoltzConfig
    | ChaiConfig
    | ProtenixConfig
)


def dump_config(conf: AnyConfig, fmt: str = "yaml", **kwargs) -> str:
    """Serialize a config model to a string.

    Args:
        conf: The config model to serialize.
        fmt: Output format (``"yaml"`` or ``"json"``).
        **kwargs: Extra keyword arguments forwarded to ``model_dump`` /
            ``model_dump_json``.

    Returns:
        The serialized config string.

    Raises:
        ValueError: If *fmt* is not ``"yaml"`` or ``"json"``.

    """
    if fmt == "yaml":
        return conf.to_yaml(**kwargs)
    elif fmt == "json":
        return conf.to_json(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'yaml' or 'json'.")


def write_config(conf: AnyConfig, out_file: str | Path, **kwargs) -> None:
    """Write a config model to a file.

    The output format is inferred from the file extension.

    Args:
        conf: The config model to write.
        out_file: Output file path (``.yaml``, ``.yml``, or ``.json``).
        **kwargs: Extra keyword arguments forwarded to :func:`dump_config`.

    Raises:
        ValueError: If the file extension is unsupported.

    """
    suffix = Path(out_file).suffix.lower()
    if suffix in {".yml", ".yaml"}:
        fmt = "yaml"
    elif suffix == ".json":
        fmt = "json"
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")

    text = dump_config(conf, fmt=fmt, **kwargs)
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
