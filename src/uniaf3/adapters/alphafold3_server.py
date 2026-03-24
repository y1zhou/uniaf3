"""Adapter for converting between UniAF3Config and AlphaFold3 Server config."""

from __future__ import annotations

from uniaf3.adapters._helpers import (
    ensure_list,
    err_unsupported_feature,
    warn_lossy_conversion,
)
from uniaf3.constant import KNOWN_ION_CCD_CODES, KNOWN_LIGAND_CCD_CODES
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
    AuxiliaryParams,
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    SequenceModification,
    UniAF3Config,
)
from uniaf3.utils import int_to_letters


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
    warn_lossy_conversion(
        "UniAF3Config.sequences[*].id are converted to AF3ServerSequenceEntry.*.count; explicit chain IDs are not preserved."
    )
    sequences: list[AF3ServerSequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, Glycan):
            # TODO: AF3 Server represents glycans as modifications on protein
            # chains, not standalone glycan entities. We need to find the
            # corresponding covalent bond field and merge them into the
            # protein chain entry
            err_unsupported_feature(
                strict,
                f"Standalone glycans are not directly supported in AF3 Server: {seq}",
            )
            continue

        if isinstance(seq, ProteinSeq) or (
            isinstance(seq, Polymer) and seq.polymer_type == PolymerType.Protein
        ):
            ids = ensure_list(seq.id)
            if seq.description is not None:
                warn_lossy_conversion(
                    "UniAF3Config.sequences[*].description is not represented in AF3ServerSequenceEntry.proteinChain."
                )
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerProteinModification(
                        ptmType=f"CCD_{m.ccd}", ptmPosition=m.position
                    )
                    for m in seq.modifications
                ]

            if isinstance(seq, ProteinSeq) and (
                seq.unpaired_msa is not None
                or seq.paired_msa is not None
                or seq.templates is not None
            ):
                warn_lossy_conversion(
                    "UniAF3 ProteinSeq fields {unpaired_msa,paired_msa,templates} are not represented in AF3ServerSequenceEntry.proteinChain."
                )
            protein = AF3ServerProtein(
                sequence=seq.sequence,
                count=len(ids),
                modifications=mods,
            )
            sequences.append(AF3ServerSequenceEntry(proteinChain=protein))

        elif isinstance(seq, Polymer) and seq.polymer_type == PolymerType.DNA:
            ids = ensure_list(seq.id)
            if seq.description is not None:
                warn_lossy_conversion(
                    "UniAF3Config.sequences[*].description is not represented in AF3ServerSequenceEntry.dnaSequence."
                )
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerDNAModification(
                        modificationType=f"CCD_{m.ccd}", basePosition=m.position
                    )
                    for m in seq.modifications
                ]
            dna = AF3ServerDNA(
                sequence=seq.sequence, count=len(ids), modifications=mods
            )
            sequences.append(AF3ServerSequenceEntry(dnaSequence=dna))

        elif isinstance(seq, Polymer) and seq.polymer_type == PolymerType.RNA:
            ids = ensure_list(seq.id)
            if seq.description is not None:
                warn_lossy_conversion(
                    "UniAF3Config.sequences[*].description is not represented in AF3ServerSequenceEntry.rnaSequence."
                )
            mods = None
            if seq.modifications:
                mods = [
                    AF3ServerRNAModification(
                        modificationType=f"CCD_{m.ccd}", basePosition=m.position
                    )
                    for m in seq.modifications
                ]
            rna = AF3ServerRNA(
                sequence=seq.sequence, count=len(ids), modifications=mods
            )
            sequences.append(AF3ServerSequenceEntry(rnaSequence=rna))

        elif isinstance(seq, Ligand):
            ids = ensure_list(seq.id)
            count = len(ids)
            if seq.ccd is not None:
                if len(seq.ccd) > 1:
                    warn_lossy_conversion(
                        f"UniAF3Config.sequences[*].Ligand.ccd supports multiple entries, but AF3ServerSequenceEntry.ligand.ligand accepts one code; only '{seq.ccd[0]}' is kept."
                    )
                    err_unsupported_feature(
                        strict,
                        f"AF3 Server only supports one CCD code per ligand: {seq}",
                    )
                ccd_code = seq.ccd[0]
                if ccd_code in KNOWN_ION_CCD_CODES:
                    sequences.append(
                        AF3ServerSequenceEntry(
                            ion=AF3ServerIon(ion=ccd_code, count=count)
                        )
                    )
                else:
                    ligand_name = (
                        ccd_code
                        if ccd_code in KNOWN_LIGAND_CCD_CODES
                        else f"CCD_{ccd_code}"
                    )
                    if ligand_name in KNOWN_LIGAND_CCD_CODES:
                        sequences.append(
                            AF3ServerSequenceEntry(
                                ligand=AF3ServerLigand(ligand=ligand_name, count=count)
                            )
                        )
                    else:
                        err_unsupported_feature(
                            strict,
                            f"Unsupported ligand CCD code for AF3 Server: {ccd_code}",
                        )
            elif seq.smiles:
                err_unsupported_feature(
                    strict,
                    f"AF3 Server only accepts CCD ligands, not SMILES: {seq}",
                )

    for field in (
        config.covalent_bonds,
        config.contact_restraints,
        config.pocket_restraints,
    ):
        if field is not None:
            err_unsupported_feature(
                strict, f"AF3 Server does not support constraints: {field}"
            )
    return AF3ServerJob(name=name, modelSeeds=config.aux.seeds, sequences=sequences)


def to_alphafold3_server(
    config: UniAF3Config | list[UniAF3Config],
    name: str = "uniaf3_job",
    strict: bool = False,
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

    # AF3 Server uses count-based copies, not chain IDs.
    # We generate chain IDs based on entity order (A, B, C, ...).
    chain_counter = 0

    def _next_chain_ids(count: int) -> str | list[str]:
        nonlocal chain_counter
        ids = []
        for _ in range(count):
            ids.append(int_to_letters(chain_counter + 1))
            chain_counter += 1
        return ids[0] if len(ids) == 1 else ids

    for entry in job.sequences:
        if entry.proteinChain is not None:
            pc = entry.proteinChain
            chain_ids = _next_chain_ids(pc.count)
            if pc.glycans is not None:
                warn_lossy_conversion(
                    "AF3ServerSequenceEntry.proteinChain.glycans are not converted to UniAF3Config.sequences and are dropped."
                )
            if pc.maxTemplateDate is not None or pc.useStructureTemplate is not True:
                warn_lossy_conversion(
                    "AF3ServerSequenceEntry.proteinChain.{useStructureTemplate,maxTemplateDate} are not represented in UniAF3Config."
                )
            mods = None
            if pc.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.ptmType.removeprefix("CCD_"), position=m.ptmPosition
                    )
                    for m in pc.modifications
                ]
            seq = ProteinSeq(
                polymer_type=PolymerType.Protein,
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
                polymer_type=PolymerType.DNA,
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
                polymer_type=PolymerType.RNA,
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

    # AF3 Server config does not include seeds; default to [42].
    if not job.modelSeeds:
        warn_lossy_conversion(
            "AF3ServerJob.modelSeeds is empty; UniAF3Config.aux.seeds defaults to [42]."
        )
    return UniAF3Config(
        sequences=sequences,
        aux=AuxiliaryParams(seeds=job.modelSeeds if job.modelSeeds else [42]),
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

    return [_from_alphafold3_server(job) for job in config]  # ty:ignore[not-iterable]
