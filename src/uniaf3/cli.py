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


def _load_config(path: Path, fmt: str):
    """Load and validate a config file based on format."""
    import json

    import yaml

    if fmt == "uniaf3":
        from uniaf3.schema import UniAF3Config

        return UniAF3Config.from_file(path)
    elif fmt == "alphafold3":
        from uniaf3.schema.alphafold3 import AF3Config

        data = json.loads(path.read_text())
        return AF3Config.model_validate(data)
    elif fmt == "boltz":
        from uniaf3.schema.boltz import BoltzConfig

        with open(path) as f:
            data = yaml.safe_load(f)
        return BoltzConfig.model_validate(data)
    elif fmt == "chai":
        from uniaf3.schema.chai import ChaiConfig

        with open(path) as f:
            data = yaml.safe_load(f)
        return ChaiConfig.model_validate(data)
    elif fmt == "protenix":
        import json

        from uniaf3.schema.protenix import ProtenixConfig

        data = json.loads(path.read_text())
        # Protenix top-level is a list of jobs
        if isinstance(data, list):
            return ProtenixConfig.model_validate({"jobs": data})
        return ProtenixConfig.model_validate(data)
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
    """Validate an input config file and print the corresponding modal run command."""
    try:
        conf = _load_config(input_config_file, format.value)
    except Exception as exc:
        console.print(f"[bold red]Validation error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if format.value == "uniaf3":
        console.print(f"Config hash: [bold]{conf.hash}[/bold]")

    console.print("[bold green]Config is valid![/bold green]\n")

    # Print the config in the appropriate format
    if hasattr(conf, "yaml_str"):
        console.print(Syntax(conf.yaml_str, "yaml", theme="one-dark"))
    elif hasattr(conf, "json_str"):
        console.print(Syntax(conf.json_str, "json", theme="one-dark"))


def _to_uniaf3(conf, fmt: str):
    """Convert any model config to UniAF3Config."""
    if fmt == "uniaf3":
        return conf
    elif fmt == "alphafold3":
        from uniaf3.adapters import from_alphafold3

        return from_alphafold3(conf)
    elif fmt == "boltz":
        from uniaf3.adapters import from_boltz

        return from_boltz(conf)
    elif fmt == "chai":
        from uniaf3.adapters import from_chai

        return from_chai(conf)
    elif fmt == "protenix":
        from uniaf3.adapters import from_protenix

        return from_protenix(conf)
    else:
        raise ValueError(f"Cannot convert from format: {fmt}")


def _from_uniaf3(uni_conf, fmt: str, name: str = "uniaf3_job"):
    """Convert a UniAF3Config to a model-specific config."""
    if fmt == "uniaf3":
        return uni_conf
    elif fmt == "alphafold3":
        from uniaf3.adapters import to_alphafold3

        return to_alphafold3(uni_conf, name=name)
    elif fmt == "boltz":
        from uniaf3.adapters import to_boltz

        return to_boltz(uni_conf)
    elif fmt == "chai":
        from uniaf3.adapters import to_chai

        return to_chai(uni_conf)
    elif fmt == "protenix":
        from uniaf3.adapters import to_protenix

        return to_protenix(uni_conf, name=name)
    else:
        raise ValueError(f"Cannot convert to format: {fmt}")


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
    from uniaf3.schema import write_config

    try:
        src_conf = _load_config(input_config_file, from_format.value)
        uni_conf = _to_uniaf3(src_conf, from_format.value)
        dst_conf = _from_uniaf3(uni_conf, to_format.value, name=input_config_file.stem)
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
