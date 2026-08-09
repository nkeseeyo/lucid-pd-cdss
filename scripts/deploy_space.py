"""Assemble and publish the LUCID-PD Hugging Face Space.

A Space is a self-contained git repository, so the parts of this project it needs are
first collected into one staging tree and then uploaded. Staging is deterministic: the
tree is rebuilt from scratch on every run, and only the files listed in ``LAYOUT`` are
copied, so nothing incidental in the working directory reaches the published Space.

    python scripts/deploy_space.py --repo nkeseeyo/lucid-pd-cdss
    python scripts/deploy_space.py --repo nkeseeyo/lucid-pd-cdss --stage-only
    python scripts/deploy_space.py --repo nkeseeyo/lucid-pd-cdss --private

Authentication uses the token passed with ``--token``, otherwise ``HF_TOKEN`` from the
environment, otherwise the credential stored by ``hf auth login``. The token must be a
write token belonging to the account that owns the target repository, and that account
must hold a paid plan, since Hugging Face permits only static Spaces on free personal
accounts and rejects the creation of a Docker Space with HTTP 402.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_SOURCE = PROJECT_ROOT / "deploy" / "hf_space"
STAGING = PROJECT_ROOT / "build" / "hf_space"

#: (source, destination) pairs relative to the project root and the staging tree. A source
#: directory is copied recursively, filtered by IGNORED; a source file is copied as it is.
LAYOUT: tuple[tuple[Path, str], ...] = (
    (SPACE_SOURCE / "Dockerfile", "Dockerfile"),
    (SPACE_SOURCE / "README.md", "README.md"),
    (SPACE_SOURCE / "requirements.txt", "requirements.txt"),
    (SPACE_SOURCE / ".dockerignore", ".dockerignore"),
    (PROJECT_ROOT / "src" / "pdcdss", "src/pdcdss"),
    (PROJECT_ROOT / "app" / "backend", "app/backend"),
    (PROJECT_ROOT / "app" / "frontend" / "src", "frontend/src"),
    (PROJECT_ROOT / "app" / "frontend" / "public", "frontend/public"),
    (PROJECT_ROOT / "app" / "frontend" / "index.html", "frontend/index.html"),
    (PROJECT_ROOT / "app" / "frontend" / "package.json", "frontend/package.json"),
    (PROJECT_ROOT / "app" / "frontend" / "package-lock.json", "frontend/package-lock.json"),
    (PROJECT_ROOT / "app" / "frontend" / "tsconfig.json", "frontend/tsconfig.json"),
    (PROJECT_ROOT / "app" / "frontend" / "vite.config.ts", "frontend/vite.config.ts"),
    (PROJECT_ROOT / "models" / "deployed_voice.joblib", "models/deployed_voice.joblib"),
    (PROJECT_ROOT / "models" / "deployed_mri.pt", "models/deployed_mri.pt"),
    (PROJECT_ROOT / "data" / "external" / "guidance", "data/external/guidance"),
)

#: Build products and caches: the image regenerates them, and shipping them would make the
#: upload far larger than the code it carries.
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist",
                                 ".DS_Store", "*.egg-info")


def stage() -> Path:
    """Rebuild the staging tree and return its path, reporting anything missing."""
    missing = [source for source, _ in LAYOUT if not source.exists()]
    if missing:
        listed = "\n  ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        sys.exit(f"cannot stage the Space, these inputs are absent:\n  {listed}\n"
                 "Rebuild the model artefacts with `python -m pdcdss.speech.deploy_voice` "
                 "and `python -m pdcdss.mri.deploy_mri`.")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    for source, destination in LAYOUT:
        target = STAGING / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORED)
        else:
            shutil.copy2(source, target)
    return STAGING


def _describe(paths: Iterable[Path]) -> str:
    files = [path for path in paths if path.is_file()]
    megabytes = sum(path.stat().st_size for path in files) / 1_048_576
    return f"{len(files)} files, {megabytes:.1f} MB"


def _check_owner(api, repo_id: str) -> None:
    """Stop before the upload if the credential cannot own a Docker Space.

    Hugging Face allows only static Spaces on free personal accounts, and rejects the
    creation of anything that runs on compute with HTTP 402. Reading the account first
    turns that into a message that names the cause instead of a failure mid-publish.
    """
    owner = repo_id.split("/")[0]
    identity = api.whoami()
    if identity.get("name") != owner:
        return
    if not identity.get("isPro"):
        sys.exit(f"the account '{owner}' is on the free plan, which may host only static "
                 "Spaces. Creating a Docker Space needs PRO for a personal account, or a "
                 "Team or Enterprise plan for an organisation. Subscribe at "
                 "https://huggingface.co/pro, or pass --repo for an account that qualifies.")


def publish(repo_id: str, *, token: str | None, private: bool) -> str:
    """Create the Space if it does not exist and upload the staged tree into it."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _check_owner(api, repo_id)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(STAGING),
                      commit_message="Deploy LUCID-PD CDSS")
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True,
                        help="target Space, as owner/name (e.g. nkeseeyo/lucid-pd-cdss)")
    parser.add_argument("--token", default=None,
                        help="Hugging Face write token; defaults to the stored credential")
    parser.add_argument("--private", action="store_true",
                        help="create the Space privately instead of publicly")
    parser.add_argument("--stage-only", action="store_true",
                        help="build the staging tree and stop, without uploading")
    args = parser.parse_args()

    folder = stage()
    print(f"staged {_describe(folder.rglob('*'))} -> {folder}")
    if args.stage_only:
        print("stage-only requested, nothing uploaded")
        return

    url = publish(args.repo, token=args.token, private=args.private)
    print(f"uploaded -> {url}")
    print("The Space builds the image on push; watch the Logs tab until it reads Running.")
    print("Set HF_TOKEN as a Space secret to enable the generated explanation "
          "(Settings -> Variables and secrets).")


if __name__ == "__main__":
    main()
