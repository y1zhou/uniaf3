"""Helper script for constructing actual modal run commands."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uniaf3.adapters import AnyConfig
from uniaf3.schema import UniAF3Config

app = typer.Typer(pretty_exceptions_short=False)
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
    from uniaf3.schema import ChaiConfig

    parser_map = _get_format_to_config()
    parser = parser_map.get(fmt)
    if parser is None:
        raise ValueError(f"Unknown format: {fmt}")

    # Special treatment for Chai since it uses FASTA and optional restraints CSV
    if parser is ChaiConfig:
        if path.suffix in {".yaml", ".yml"}:
            return parser.from_yaml(path)

        for suffix in (".restraints", ".csv"):
            restraints_path = path.with_suffix(suffix)
            if restraints_path.exists():
                return parser.from_file(path, restraints_path)

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
        console.print_exception(show_locals=True, width=console.width)
        console.print(
            f"[bold yellow]Validation error:[/bold yellow] {input_config_file}"
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print_exception(show_locals=True, width=console.width)
        console.print(f"[bold red]Error loading config:[/bold red] {input_config_file}")
        raise typer.Exit(code=1) from exc

    if format.value == "uniaf3":
        if not isinstance(conf, UniAF3Config):
            raise TypeError(f"Expected UniAF3Config, got {type(conf)}")
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
        uni_conf = to_uniaf3(src_conf, msa_dir=output_dir / "msa")
        parser_map = _get_format_to_config()
        parser = parser_map.get(to_format.value)
        if parser is None:
            raise ValueError(f"Unknown output format: {to_format.value}")

        dst_conf = from_uniaf3(
            uni_conf, parser, name=input_config_file.stem, msa_dir=output_dir / "msa"
        )
        if prefix is None:
            prefix = input_config_file.stem
        if isinstance(dst_conf, AnyConfig):
            dst_conf.to_files(output_dir, prefix)
        elif isinstance(dst_conf, list):
            for i, conf in enumerate(dst_conf):
                conf.to_files(output_dir, f"{prefix}_{i}")
        else:
            raise TypeError(f"Unexpected output config type: {type(dst_conf)}")
        console.print(
            f"[bold green]Converted {from_format.value} → {to_format.value}[/bold green]"
        )
        console.print(f"Outputs generated under directory: {output_dir}")
    except Exception as exc:
        console.print_exception(show_locals=True, width=console.width)
        console.print(f"[bold red]Conversion error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="msa")
def add_msa(
    input_config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the input config file.",
            exists=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Output directory for the config file(s) with MSA paths."),
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
    msa_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--msa-cache-dir",
            help="Directory to cache MSA files. Defaults to $XDG_CACHE_HOME/uniaf3/colabfold_msas/.",
        ),
    ] = None,
    chains: Annotated[
        str | None,
        typer.Option(
            "--chains",
            "-c",
            help="Comma-separated chain IDs to query MSAs for. If not set, all protein chains are processed.",
        ),
    ] = None,
    search_templates: Annotated[
        bool,
        typer.Option(
            "--search-templates/--no-search-templates",
            help="Whether to search for structural templates.",
        ),
    ] = False,
    num_templates: Annotated[
        int,
        typer.Option(
            "--num-templates",
            help="Number of templates to fetch per sequence.",
        ),
    ] = 5,
    template_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--template-cache-dir",
            help="Directory to cache template files. Defaults to $XDG_CACHE_HOME/uniaf3/rcsb/.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-query MSAs even if cached results exist.",
        ),
    ] = False,
    with_uniaf3: Annotated[
        bool,
        typer.Option(
            "--with-uniaf3",
            help="Also write a UniAF3 YAML file alongside the output format.",
        ),
    ] = False,
) -> None:
    """Query MSAs for protein sequences and write config files with MSA paths."""
    from uniaf3.adapters import from_uniaf3, to_uniaf3
    from uniaf3.schema import UniAF3Config

    try:
        # 1. Load config
        src_conf = _load_config(input_config_file, format.value)

        # 2. Convert to UniAF3 (may return a list for multi-job configs)
        if not isinstance(src_conf, UniAF3Config):
            uni_confs = to_uniaf3(src_conf)
        else:
            uni_confs = src_conf
        if not isinstance(uni_confs, list):
            uni_confs = [uni_confs]

        # 3. Parse chains argument
        chain_set = set(chains.split(",")) if chains else None

        # 4. Process each config sequentially
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, uni_conf in enumerate(uni_confs):
            uni_conf.add_msa_for_protein_seqs(
                msa_cache_dir=msa_cache_dir,
                chains=chain_set,
                search_templates=search_templates,
                num_templates_per_seq=num_templates,
                template_cache_dir=template_cache_dir,
                force=force,
            )

            # Determine prefix: use input stem for single config, append index for multiple
            prefix = (
                input_config_file.stem
                if len(uni_confs) == 1
                else f"{input_config_file.stem}_{i}"
            )

            # Write in original format (convert back if needed)
            if format.value != "uniaf3":
                parser = _get_format_to_config()[format.value]
                msa_out_dir = output_dir / "msa"
                dst_conf = from_uniaf3(
                    uni_conf, parser, name=prefix, msa_dir=msa_out_dir
                )
                if isinstance(dst_conf, list):
                    for j, dc in enumerate(dst_conf):
                        dc.to_files(output_dir, f"{prefix}_{j}")
                else:
                    dst_conf.to_files(output_dir, prefix)
            else:
                uni_conf.to_files(output_dir, prefix)

            # Optionally also write UniAF3 YAML
            if with_uniaf3 and format.value != "uniaf3":
                uni_conf.to_files(output_dir, f"{prefix}_uniaf3")

            console.print(f"[green]Processed config {i + 1}/{len(uni_confs)}[/green]")

        console.print(
            f"[bold green]MSA query complete.[/bold green] "
            f"Outputs written to: {output_dir}"
        )
    except Exception as exc:
        console.print_exception(show_locals=True, width=console.width)
        console.print(f"[bold red]MSA query error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
