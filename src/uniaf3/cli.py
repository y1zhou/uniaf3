"""Helper script for constructing actual modal run commands."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uniaf3.adapters import AnyConfig

app = typer.Typer()
console = Console(tab_size=2)


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """UniAF3: Unified AlphaFold3-family config toolkit.

    Validate, convert, and inspect input configs for AlphaFold3, Boltz,
    Chai-1, Protenix, and the unified UniAF3 format.
    """


##########################################
# CLI Commands
##########################################
class ConfigFormat(StrEnum):
    """Supported input config formats."""

    UniAF3 = "uniaf3"
    AlphaFold3 = "alphafold3"
    AF3 = "alphafold3"
    Boltz = "boltz"
    Boltz1 = "boltz"
    Boltz2 = "boltz"
    Chai = "chai"
    Chai1 = "chai"
    Protenix = "protenix"
    AlphaFold3Server = "alphafold3server"


def _load_config(path: Path, fmt: str) -> AnyConfig:
    """Load and validate a config file using the schema's from_file method.

    Args:
        path: Path to the config file.
        fmt: The config format identifier.

    Returns:
        A validated config object.

    Raises:
        ValueError: If the format is unknown.

    """
    if fmt == "uniaf3":
        from uniaf3.schema import UniAF3Config

        return UniAF3Config.from_file(path)
    elif fmt == "alphafold3":
        from uniaf3.schema import AF3Config

        return AF3Config.from_file(path)
    elif fmt == "boltz":
        from uniaf3.schema import BoltzConfig

        return BoltzConfig.from_file(path)
    elif fmt == "chai":
        from uniaf3.schema import ChaiConfig

        return ChaiConfig.from_file(path)
    elif fmt == "protenix":
        from uniaf3.schema import ProtenixConfig

        return ProtenixConfig.from_file(path)
    elif fmt == "alphafold3server":
        from uniaf3.schema import AF3ServerConfig

        return AF3ServerConfig.from_file(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


@app.command(name="validate")
def validate_config(
    input_config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to a UniAF3 input config file to validate.",
            exists=True,
            resolve_path=True,
        ),
    ],
    format: Annotated[
        ConfigFormat,
        typer.Option(
            "--format",
            "-f",
            help="Format of the input config file",
            case_sensitive=False,
        ),
    ] = ConfigFormat.UniAF3,
) -> None:
    """Validate an input config file and print its contents."""
    try:
        conf = _load_config(input_config_file, format.value)
    except Exception as exc:
        console.print(f"[bold red]Validation error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if format.value == "uniaf3":
        from uniaf3.schema import UniAF3Config

        assert isinstance(conf, UniAF3Config)
        console.print(f"Config hash: [bold]{conf.hash}[/bold]")

    console.print(
        f"[bold green]{type(conf).__name__}[/bold green] config is valid! "
        "Dictionary representation displayed below:\n"
    )

    # Print the config in the appropriate format
    console.print(conf.model_dump())


def _get_format_to_config():
    """Lazily build format-to-config-class mapping."""
    from uniaf3.schema import (
        AF3Config,
        AF3ServerConfig,
        BoltzConfig,
        ChaiConfig,
        ProtenixConfig,
        UniAF3Config,
    )

    return {
        "uniaf3": UniAF3Config,
        "alphafold3": AF3Config,
        "alphafold3server": AF3ServerConfig,
        "boltz": BoltzConfig,
        "chai": ChaiConfig,
        "protenix": ProtenixConfig,
    }


@app.command(name="convert")
def convert_config(
    input_config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the input config file.", exists=True, resolve_path=True
        ),
    ],
    output_config_file: Annotated[
        Path, typer.Argument(help="Path to the output config file.")
    ],
    from_format: Annotated[
        ConfigFormat,
        typer.Option(
            "--from-format",
            "-f",
            help="Format of the input config file",
            case_sensitive=False,
        ),
    ] = ConfigFormat.UniAF3,
    to_format: Annotated[
        ConfigFormat,
        typer.Option(
            "--to-format",
            "-t",
            help="Format of the output config file",
            case_sensitive=False,
        ),
    ] = ConfigFormat.AlphaFold3,
):
    """Convert an input config file from one format to another."""
    from uniaf3.adapters import from_uniaf3, to_uniaf3
    from uniaf3.schema import write_config

    try:
        src_conf = _load_config(input_config_file, from_format.value)
        uni_conf = to_uniaf3(src_conf)
        fmt_map = _get_format_to_config()
        dst_conf = from_uniaf3(
            uni_conf, fmt_map[to_format.value], name=input_config_file.stem
        )
        write_config(dst_conf, output_config_file)
        console.print(
            f"[bold green]Converted {from_format.value} → {to_format.value}[/bold green]"
        )
        console.print(f"Output: {output_config_file}")
    except Exception as exc:
        console.print(f"[bold red]Conversion error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
