"""Resolve every directory the BCG code reads and writes.

Nothing in this repository stores an absolute path. A script asks this module
where the dataset, the workspace and the checkpoint are, and this module reads
that from ``config.yaml`` beside it.

To set the repository up:

    cp config.example.yaml config.yaml     # then edit the two required paths

``BCG_CONFIG`` overrides the location of the file itself, and the environment
variables named in ``_ENV_OVERRIDES`` override individual entries, which is
convenient on a cluster where the paths come from a job script.

Layout created under ``workspace_root``. The scripts create these as they go;
none of them has to exist beforehand:

    masks/train_reconstructed/   crack-free layouts rebuilt from the training masks
    masks/test_reconstructed/    the same reconstruction for the test partition
    masks/test_annotations/      the rasterised test annotations
    runs/crack_unet/             the crack-probability U-Net training run
    evaluation/                  the paired learned-prior path evaluation
    analysis/<name>/             every other analysis output
"""

from __future__ import annotations

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE_FILE = REPO_ROOT / "config.example.yaml"

_ENV_OVERRIDES = {
    "dataset_root": "BCG_DATASET_ROOT",
    "workspace_root": "BCG_WORKSPACE",
    "unet_checkpoint": "BCG_UNET_CHECKPOINT",
}

_REQUIRED = ("dataset_root", "workspace_root")

_MISSING_FILE = """\
No BCG configuration found.

Copy the example and edit the two required paths:

    cp {example} {target}

Then set `dataset_root` to your annotated masonry dataset and `workspace_root`
to a directory the scripts may create and write to. `BCG_CONFIG` points at a
configuration file somewhere else."""


def _config_path() -> Path:
    override = os.environ.get("BCG_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "config.yaml"


def _read(path: Path) -> dict:
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level.")
    return loaded


def _resolve(value, base: Path) -> Path:
    """Expand a configured path; a relative one is taken from the repository."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


class Paths:
    """The directories a script needs, resolved once on first use."""

    def __init__(self) -> None:
        self._values: dict | None = None

    # -- loading ---------------------------------------------------------
    def _load(self) -> dict:
        if self._values is not None:
            return self._values

        path = _config_path()
        values: dict = {}
        if path.exists():
            values = _read(path)
        elif not all(os.environ.get(_ENV_OVERRIDES[k]) for k in _REQUIRED):
            raise FileNotFoundError(
                _MISSING_FILE.format(example=EXAMPLE_FILE.name, target=path.name)
            )

        for key, variable in _ENV_OVERRIDES.items():
            if os.environ.get(variable):
                values[key] = os.environ[variable]

        missing = [k for k in _REQUIRED if not values.get(k)]
        if missing:
            raise KeyError(
                f"{path} does not set {', '.join(missing)}. "
                f"See {EXAMPLE_FILE.name} for what each entry means."
            )

        self._values = values
        return values

    def get(self, key, default=None):
        """Return a raw configured value, for entries that are not paths."""
        return self._load().get(key, default)

    # -- the two configured roots ----------------------------------------
    @property
    def dataset_root(self) -> Path:
        """The annotated masonry dataset, holding train/, valid/ and test/."""
        return _resolve(self._load()["dataset_root"], REPO_ROOT)

    @property
    def workspace(self) -> Path:
        """Everything the scripts derive and write."""
        return _resolve(self._load()["workspace_root"], REPO_ROOT)

    @property
    def checkpoint(self) -> Path:
        """The crack-probability U-Net checkpoint."""
        configured = self._load().get("unet_checkpoint")
        if not configured:
            return self.unet_run / "best.pt"
        return _resolve(configured, REPO_ROOT)

    # -- derived locations -----------------------------------------------
    @property
    def train_masks(self) -> Path:
        """Crack-free layouts reconstructed from the training masks."""
        return self.workspace / "masks/train_reconstructed"

    @property
    def test_masks(self) -> Path:
        """The same reconstruction for the test partition."""
        return self.workspace / "masks/test_reconstructed"

    @property
    def test_annotations(self) -> Path:
        """The rasterised test annotations."""
        return self.workspace / "masks/test_annotations"

    @property
    def unet_run(self) -> Path:
        """The crack-probability U-Net training run."""
        return self.workspace / "runs/crack_unet"

    @property
    def prior_evaluation(self) -> Path:
        """The paired learned-prior path evaluation."""
        return self.workspace / "evaluation"

    def analysis(self, name: str) -> Path:
        """Any other analysis output directory."""
        return self.workspace / "analysis" / name

    # -- code inside this repository -------------------------------------
    @staticmethod
    def unet_code_dir() -> Path:
        """The directory holding ``models.py`` for the crack-probability U-Net."""
        return REPO_ROOT / "01_crack_path_generation/unet"

    @staticmethod
    def layout_code_dir() -> Path:
        """The directory holding ``procedural_layout.py``."""
        return REPO_ROOT / "01_crack_path_generation"


paths = Paths()


def describe() -> str:
    """Print the resolved locations, so a setup can be checked before a run."""
    lines = [
        f"configuration    {_config_path()}",
        f"repository       {REPO_ROOT}",
        f"dataset_root     {paths.dataset_root}",
        f"workspace_root   {paths.workspace}",
        f"unet_checkpoint  {paths.checkpoint}",
        "",
        f"train masks      {paths.train_masks}",
        f"test masks       {paths.test_masks}",
        f"test annotations {paths.test_annotations}",
        f"U-Net run        {paths.unet_run}",
        f"evaluation      {paths.prior_evaluation}",
    ]
    missing = [
        name
        for name, value in (
            ("dataset_root", paths.dataset_root),
            ("unet_checkpoint", paths.checkpoint),
        )
        if not value.exists()
    ]
    lines.append("")
    lines.append(
        "all configured inputs exist"
        if not missing
        else "does not exist yet: " + ", ".join(missing)
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
