"""Utility functions for UniAF3."""

import hashlib
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio


def hash_sequence(seq: str | bytes) -> str:
    """Compute the Chai-style sequence hash.

    Source: chai_lab.data.parsing.msas.aligned_pqt.hash_sequence
    """
    return hashlib.sha256(seq.encode() if isinstance(seq, str) else seq).hexdigest()


def int_to_letters(n: int) -> str:
    """Convert int to letters.

    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number

    Returns:
        chain ID, e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB

    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def download_file(
    session: aiohttp.ClientSession, url: str, local_path: Path, chunk_size: int = 8192
):
    """Download a file asynchronously using aiohttp."""
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            not local_path.exists()
            or response.content_length != local_path.stat().st_size
        ):
            with open(local_path, "wb") as f:
                while chunk := await response.content.read(chunk_size):
                    f.write(chunk)


async def download_files(
    urls: dict[str, str | Path],
    force: bool = False,
    max_connections: int = 50,
    num_retries: int = 1,
    progress_bar_desc: str | None = None,
):
    """Download multiple files concurrently using aiohttp.

    Args:
        urls: Keys are URLs, and values are local file paths.
        force: Whether to overwrite existing files.
        max_connections: Limit concurrent downloads to be civil.
        num_retries: Number of times to retry failed downloads.
        progress_bar_desc: Optional description for the progress bar.

    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    }

    # launch downloads concurrently
    connector = aiohttp.TCPConnector(limit=max_connections)
    for attempt in range(num_retries):
        try:
            async with aiohttp.ClientSession(
                headers=headers, connector=connector
            ) as session:
                tasks = []
                for url, local_file in urls.items():
                    local_path = Path(local_file)
                    if force or not local_path.exists():
                        tasks.append(download_file(session, url, local_path))

                # run all of the downloads and await their completion
                await tqdm_asyncio.gather(*tasks, desc=progress_bar_desc)
        except Exception as e:
            print(f"Download attempt failed with error: {e}")
            if attempt == num_retries - 1:
                raise RuntimeError("All download attempts failed.") from e
            else:
                print(f"Retrying downloads (attempt {attempt + 2}/{num_retries})...")
