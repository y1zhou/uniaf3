"""Adapter for converting between UniAF3Config and AlphaFold3 Server config."""

from __future__ import annotations

from uniaf3.adapters._helpers import (
    _KNOWN_ION_CCD_CODES,
    _ensure_list,
    err_unsupported_feature,
)
from uniaf3.schema.alphafold3_server import (
    AF3ServerConfig,
    AF3ServerDNA,
    AF3ServerDNAModification,
    AF3ServerIon,
    AF3ServerJob,
    AF3ServerLigand,
    AF3ServerProtein,
    AF3ServerProteinModification,
    AF3ServerRNA,
    AF3ServerRNAModification,
    AF3ServerSequenceEntry,
)
from uniaf3.schema.base import (
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    SequenceModification,
    UniAF3Config,
)


def _to_alphafold3_server(
    config: UniAF3Config, name: str = "uniaf3_job", strict: bool = True
) -> AF3ServerJob:
    """Convert a UniAF3Config to an AlphaFold3 Server job.

    The server config is simpler — no MSA, no templates, no userCCD.
    Ions (detected from known CCD codes) get their own entity type.

    Args:
        config: UniAF3Config pydantic object.
        name: Job name for the AF3 Server job.
        strict: If True, raise errors when encountering unsupported features.

    """
    sequences: list[AF3ServerSequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, Glycan):
            # NOTE: AF3 Server represents glycans as modifications on protein
            # chains, not standalone entities. Cannot convert standalone glycans.
            err_unsupported_feature(
                strict,
                f"Glycans are not directly supported in AF3 Server: {seq}",
            )
            continue

        if isinstance(seq, ProteinSeq) or (
            isinstance(seq, Polymer) and seq.seq_type == PolymerType.Protein
        ):
            ids = _ensure_list(seq.id)
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerProteinModification(
                        ptmType=f"CCD_{m.ccd}", ptmPosition=m.position
                    )
                    for m in seq.modifications
                ]
            protein = AF3ServerProtein(
                sequence=seq.sequence,
                count=len(ids),
                modifications=mods,
            )
            sequences.append(AF3ServerSequenceEntry(proteinChain=protein))

        elif isinstance(seq, Polymer) and seq.seq_type == PolymerType.DNA:
            ids = _ensure_list(seq.id)
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerDNAModification(
                        modificationType=f"CCD_{m.ccd}", basePosition=m.position
                    )
                    for m in seq.modifications
                ]
            dna = AF3ServerDNA(
                sequence=seq.sequence,
                count=len(ids),
                modifications=mods,
            )
            sequences.append(AF3ServerSequenceEntry(dnaSequence=dna))

        elif isinstance(seq, Polymer) and seq.seq_type == PolymerType.RNA:
            ids = _ensure_list(seq.id)
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerRNAModification(
                        modificationType=f"CCD_{m.ccd}", basePosition=m.position
                    )
                    for m in seq.modifications
                ]
            rna = AF3ServerRNA(
                sequence=seq.sequence,
                count=len(ids),
                modifications=mods,
            )
            sequences.append(AF3ServerSequenceEntry(rnaSequence=rna))

        elif isinstance(seq, Ligand):
            ids = _ensure_list(seq.id)
            count = len(ids)
            if seq.ccd:
                for ccd_code in seq.ccd:
                    if ccd_code in _KNOWN_ION_CCD_CODES:
                        sequences.append(
                            AF3ServerSequenceEntry(
                                ion=AF3ServerIon(ion=ccd_code, count=count)
                            )
                        )
                    else:
                        sequences.append(
                            AF3ServerSequenceEntry(
                                ligand=AF3ServerLigand(
                                    ligand=f"CCD_{ccd_code}", count=count
                                )
                            )
                        )
            elif seq.smiles:
                # NOTE: AF3 Server only accepts CCD ligands, not SMILES.
                err_unsupported_feature(
                    strict,
                    f"AF3 Server does not support SMILES ligands: {seq}",
                )

    # NOTE: AF3 Server does not support restraints/bonded atom pairs.
    return AF3ServerJob(name=name, modelSeeds=config.seeds, sequences=sequences)


def to_alphafold3_server(
    config: list[UniAF3Config], name: str = "uniaf3_job", strict: bool = True
) -> AF3ServerConfig:
    """Convert a list of UniAF3Config to an AlphaFold3 Server config.

    Args:
        config: A list of UniAF3Config pydantic objects.
        name: Job name for the AF3 Server config.
        strict: If True, raise errors when encountering unsupported features.

    """
    if isinstance(config, UniAF3Config):
        # Allow passing a single config for convenience
        config = [config]

    if len(config) == 1:
        names = [name]
    else:
        names = [f"{name}_{i}" for i in range(1, len(config) + 1)]
    return AF3ServerConfig(
        [
            _to_alphafold3_server(c, name=names[i], strict=strict)
            for i, c in enumerate(config)
        ]
    )


def _from_alphafold3_server(job: AF3ServerJob) -> UniAF3Config:
    """Convert a single AF3ServerJob to UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []

    # NOTE: AF3 Server uses count-based copies, not chain IDs.
    # We generate chain IDs based on entity order (A, B, C, ...).
    chain_counter = 0

    def _next_chain_ids(count: int) -> str | list[str]:
        nonlocal chain_counter
        ids = []
        for _ in range(count):
            n = chain_counter
            if n < 26:
                ids.append(chr(65 + n))
            else:
                left_char = chr(65 + (n - 26) % 26)
                right_char = chr(65 + (n - 26) // 26)
                ids.append(f"{left_char}{right_char}")
            chain_counter += 1
        return ids[0] if len(ids) == 1 else ids

    for entry in job.sequences:
        if entry.proteinChain is not None:
            pc = entry.proteinChain
            chain_ids = _next_chain_ids(pc.count)
            mods = None
            if pc.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.ptmType.removeprefix("CCD_"), position=m.ptmPosition
                    )
                    for m in pc.modifications
                ]
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=chain_ids,
                sequence=pc.sequence,
                modifications=mods,
            )
            sequences.append(seq)

        elif entry.dnaSequence is not None:
            ds = entry.dnaSequence
            chain_ids = _next_chain_ids(ds.count)
            mods = None
            if ds.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.modificationType.removeprefix("CCD_"),
                        position=m.basePosition,
                    )
                    for m in ds.modifications
                ]
            seq = Polymer(
                seq_type=PolymerType.DNA,
                id=chain_ids,
                sequence=ds.sequence,
                modifications=mods,
            )
            sequences.append(seq)

        elif entry.rnaSequence is not None:
            rs = entry.rnaSequence
            chain_ids = _next_chain_ids(rs.count)
            mods = None
            if rs.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.modificationType.removeprefix("CCD_"),
                        position=m.basePosition,
                    )
                    for m in rs.modifications
                ]
            seq = Polymer(
                seq_type=PolymerType.RNA,
                id=chain_ids,
                sequence=rs.sequence,
                modifications=mods,
            )
            sequences.append(seq)

        elif entry.ligand is not None:
            lg = entry.ligand
            chain_ids = _next_chain_ids(lg.count)
            ccd_code = lg.ligand.removeprefix("CCD_")
            lig = Ligand(id=chain_ids, ccd=[ccd_code])
            sequences.append(lig)

        elif entry.ion is not None:
            io = entry.ion
            chain_ids = _next_chain_ids(io.count)
            lig = Ligand(id=chain_ids, ccd=[io.ion])
            sequences.append(lig)

    # NOTE: AF3 Server config does not include seeds; default to [42].
    return UniAF3Config(
        sequences=sequences,
        seeds=job.modelSeeds if job.modelSeeds else [42],
    )


def from_alphafold3_server(config: AF3ServerConfig) -> list[UniAF3Config]:
    """Convert an AlphaFold3 Server config to a list of UniAF3Config.

    Args:
        config: AF3ServerConfig pydantic object.

    Returns:
        A list of UniAF3Config.

    """
    if len(config) == 0:
        raise ValueError("AF3ServerConfig must have at least one job.")

    return [_from_alphafold3_server(job) for job in config]
