"""Query ColabFold MSA API.

https://github.com/sokrypton/ColabFold/blob/9712f2ff262d3977d571919317e06cc96c29cd95/colabfold/colabfold.py#L68
"""

import logging
import random
import tarfile
import time
from pathlib import Path

import niquests
from tqdm import tqdm

from uniaf3 import __version__

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(
    x: str | list[str],
    prefix: str | Path,
    use_env: bool = True,
    use_filter: bool = True,
    use_templates: bool = False,
    filter: bool | None = None,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    user_agent: str = f"UniAF3/{__version__} uniaf3@y1zhou.com",
) -> tuple[list[str], Path | None]:
    """Return a block of a3m lines and optionally template hits for each of the input sequences in x.

    Note that this is the Chai-1 adaptation of the original function, which has modifications
    on how templates are returned.

    Type hints were also added, and os.path was replaced with pathlib.Path.
    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers: dict[str, str] = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') to help us debug in case of problems. This warning will become an error in the future."
        )

    def submit(seqs: list[str], mode: str, N: int = 101) -> dict[str, str]:
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        with niquests.Session(
            base_url=host_url, timeout=6.02, headers=headers
        ) as session:
            while True:
                error_count = 0
                try:
                    # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                    # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                    res = session.post(
                        f"/{submission_endpoint}", data={"q": query, "mode": mode}
                    )
                except niquests.exceptions.Timeout:
                    logger.warning(
                        "Timeout while submitting to MSA server. Retrying..."
                    )
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                    time.sleep(5)
                    if error_count > 5:
                        raise
                    continue
                break

            try:
                out = res.json()
            except ValueError:
                logger.error(f"Server didn't reply with json: {res.text}")
                out = {"status": "ERROR"}
            return out

    def status(ID: str) -> dict[str, str]:
        with niquests.Session(
            base_url=host_url, timeout=6.02, headers=headers
        ) as session:
            while True:
                error_count = 0
                try:
                    res = session.get(f"/ticket/{ID}")
                except niquests.exceptions.Timeout:
                    logger.warning(
                        "Timeout while fetching status from MSA server. Retrying..."
                    )
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                    time.sleep(5)
                    if error_count > 5:
                        raise
                    continue
                break
            try:
                out = res.json()
            except ValueError:
                logger.error(f"Server didn't reply with json: {res.text}")
                out = {"status": "ERROR"}
            return out

    def download(ID: str, path: Path):
        error_count = 0
        with niquests.Session(
            base_url=host_url, timeout=6.02, headers=headers
        ) as session:
            while True:
                try:
                    res = session.get(f"/result/download/{ID}", stream=True)
                    with path.open("wb") as f:
                        for chunk in res.iter_content():
                            f.write(chunk)
                except niquests.exceptions.Timeout:
                    logger.warning(
                        "Timeout while fetching result from MSA server. Retrying..."
                    )
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                    )
                    logger.warning(f"Error: {e}")
                    time.sleep(5)
                    if error_count > 5:
                        raise
                    continue
                break

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        use_templates = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = Path(f"{prefix}_{mode}")
    path.mkdir(parents=True, exist_ok=True)

    # call mmseqs2 api
    tar_gz_file = path / "out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    # seqs_unique = []
    # [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    seqs_unique = list(dict.fromkeys(seqs))  # >=Python 3.7
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not tar_gz_file.is_file():
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)  # noqa: S311
                    logger.info(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                    )

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)  # noqa: S311
                    logger.info(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    # if TIME > 900 and out["status"] != "COMPLETE":
                    #  # something failed on the server side, need to resubmit
                    #  N += 1
                    #  break

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [path / "pair.a3m"]
    else:
        a3m_files = [path / "uniref.a3m"]
        if use_env:
            a3m_files.append(path / "bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not a3m_file.is_file() for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)  # noqa: S202

    # templates
    template_path: Path | None = None
    if use_templates:
        # print("seq\tpdb\tcid\tevalue")
        # NOTE this section has been significantly reduced to enable Chai-1 to take m8 files
        # as a common input format, while also reducing how much we ping the server.
        template_path = path / "pdb70.m8"
        if not template_path.is_file():
            raise FileNotFoundError(
                f"Expected template hits file not found at {template_path}."
            )

    # gather a3m lines
    a3m_lines: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        update_M, M = True, N - 1  # (N-1) is not a key
        with a3m_file.open() as f:
            for line in f:
                if len(line) > 0:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines[n]) for n in Ms], template_path
