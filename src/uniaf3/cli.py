"""Helper script for constructing actual modal run commands."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """UniAF3 CLI - Unified data processing for AlphaFold3-like models.

    This CLI helps users convert between inputs for AF3-like models.
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
    """Validate an input config file and print the corresponding modal run command."""
    match format.value:
        case "uniaf3":
            from uniaf3.schema import UniAF3Config

            conf = UniAF3Config.from_file(input_config_file)
            console.print(f"Config hash: [bold]{conf.hash}[/bold]")
            console.print("[bold green]Config is valid![/bold green]\n")
            console.print(Syntax(conf.yaml_str, "yaml", theme="one-dark"))

        # TODO: Implement validation for other formats
        case "alphafold3":
            raise NotImplementedError(
                "Validation for AlphaFold3 format is not yet implemented."
            )
        case "boltz":
            raise NotImplementedError(
                "Validation for Boltz format is not yet implemented."
            )
        case "chai":
            raise NotImplementedError(
                "Validation for Chai format is not yet implemented."
            )
        case "protenix":
            raise NotImplementedError(
                "Validation for Protenix format is not yet implemented."
            )
        case _:
            console.print(f"[bold red]Error:[/bold red] Unknown format '{format}'.")
            raise typer.Exit(code=1)


@app.command(name="convert")
@app.command(name="c")
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
    # TODO: Implement config conversion between different formats
    raise NotImplementedError("Config conversion is not yet implemented.")


if __name__ == "__main__":
    app()
