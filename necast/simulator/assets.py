"""Where the simulator's own data lives, and how to find it.

Three modules resolved paths independently before this one existed, with two
near-identical private helpers and one module -- the stdpopsim catalogue --
that had neither.  That third one failed silently: run from any directory other
than the repository root it found no templates at all and reported an empty
catalogue rather than an error, so the eighteen held-out cross-species models
simply were not there.

Everything the simulator needs to run ships inside the package, in
``necast/third_party``.  A relative path is looked up in the working directory
first, so a user's own file wins, and in ``necast/third_party`` second, which
is what makes an installed copy work from anywhere.
"""

from __future__ import annotations

from pathlib import Path

# The directory this package's data lives in.  Kept as one constant so a move
# is one edit rather than a search.
ASSET_ROOT = Path(__file__).resolve().parents[1] / "third_party"


def asset_path(relative: str) -> Path:
    """Resolve a configured path to a file or directory that exists.

    Absolute paths and paths that exist relative to the working directory are
    taken as given -- a caller pointing at their own map file means it.  A
    relative path that does not exist there is looked up inside the package,
    which is where the shipped maps and templates are.  If neither resolves the
    original is returned unchanged, so the caller reports the path the user
    actually asked for rather than one this function invented.
    """

    if not relative:
        return Path("")
    candidate = Path(relative).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    packaged = ASSET_ROOT / candidate
    if packaged.exists():
        return packaged
    return candidate
