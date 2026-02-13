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
# Helper functions
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
    AF3Server = "alphafold3server"


def _get_format_to_config() -> dict[str, type[AnyConfig]]:
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
    parser_map = _get_format_to_config()
    parser = parser_map.get(fmt)
    if parser is None:
        raise ValueError(f"Unknown format: {fmt}")
    return parser.from_file(path)


##########################################
# CLI Commands
##########################################
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
    from pydantic import ValidationError

    try:
        conf = _load_config(input_config_file, format.value)

    except ValidationError as exc:
        console.print(f"[bold yellow]Validation error:[/bold yellow] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Error loading config:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if format.value == "uniaf3":
        console.print(f"Config hash: [bold]{conf.hash}[/bold]")

    console.print(
        f"[bold green]{type(conf).__name__}[/bold green] config is valid! "
        "Dictionary representation displayed below:\n"
    )

    # Print the config in the appropriate format
    console.print(conf.model_dump())


@app.command(name="convert")
def convert_config(
    input_config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the input config file.", exists=True, resolve_path=True
        ),
    ],
    output_dir: Annotated[Path, typer.Argument(help="Path to the output directory.")],
    prefix: Annotated[
        str | None,
        typer.Argument(
            help="Prefix for the output config file name(s). Defaults to the input file name without extension."
        ),
    ] = None,
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

    try:
        src_conf = _load_config(input_config_file, from_format.value)
        uni_conf = to_uniaf3(src_conf)
        parser_map = _get_format_to_config()
        parser = parser_map.get(to_format.value)
        if parser is None:
            raise ValueError(f"Unknown output format: {to_format.value}")

        dst_conf = from_uniaf3(uni_conf, parser, name=input_config_file.stem)
        dst_conf.to_files(output_dir, prefix)
        console.print(
            f"[bold green]Converted {from_format.value} → {to_format.value}[/bold green]"
        )
        console.print(f"Outputs generated under directory: {output_dir}")
    except Exception as exc:
        console.print(f"[bold red]Conversion error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
