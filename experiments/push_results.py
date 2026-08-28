"""Commit and push results back to GitHub from a cloud notebook.

Colab and Kaggle VMs are destroyed when the session ends, so anything not
pushed is lost. This module commits `runs/` and `figures/` and pushes them.

Design notes
------------
* The token is never written to `.git/config` and never printed. It is passed
  inline to a single `git push` call, and every subprocess output is scrubbed
  before being shown, so a failed push cannot leak it into notebook output
  (which Colab/Kaggle save).
* `git add -f` is used so the call still works if a path is gitignored.
* A `fetch` + `rebase` happens before pushing, because the same branch is
  usually also being committed to from a laptop. If the rebase cannot be done
  cleanly, results are pushed to a timestamped branch instead of failing --
  losing a GPU run to a merge conflict would be far worse.

Usage (in a notebook):
    from experiments.push_results import push
    push(token=TOKEN, user="peeyushagarwal2004", repo="ae496-ugp")

Usage (CLI):
    python experiments/push_results.py --user <you> --repo ae496-ugp
    # token read from $GH_TOKEN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("runs", "figures")


def _run(args, token=None, check=False, cwd=None):
    """Run a git command, scrubbing the token from anything returned."""
    p = subprocess.run(args, cwd=cwd or ROOT, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if token:
        out = out.replace(token, "***")
    if check and p.returncode != 0:
        raise RuntimeError(f"`{' '.join(a for a in args if token not in a)}` failed:\n{out}")
    return p.returncode, out.strip()


def push(token=None, user=None, repo=None, branch="main", paths=None,
         message=None, author_name=None, author_email=None, verbose=True):
    """Commit `paths` and push to github.com/<user>/<repo>.

    Returns True if something was pushed, False if there was nothing to push.
    """
    token = token or os.environ.get("GH_TOKEN")
    user = user or os.environ.get("GH_USER")
    repo = repo or os.environ.get("GH_REPO", "ae496-ugp")
    if not token or not user:
        raise SystemExit("need a GitHub token and username (args or $GH_TOKEN/$GH_USER)")

    paths = list(paths or DEFAULT_PATHS)
    existing = [p for p in paths if (ROOT / p).exists()]
    if not existing:
        print(f"nothing to push: none of {paths} exist yet")
        return False

    def say(msg):
        if verbose:
            print(msg, flush=True)

    # Identity: cloud VMs have none configured, so commit would fail.
    _run(["git", "config", "user.name", author_name or f"{user} (cloud run)"])
    _run(["git", "config", "user.email", author_email or f"{user}@users.noreply.github.com"])

    # -f because results directories may be gitignored in some checkouts.
    _run(["git", "add", "-f", *existing])

    rc, _ = _run(["git", "diff", "--cached", "--quiet"])
    if rc == 0:
        say("no changes in results — nothing to commit")
        return False

    _, stat = _run(["git", "diff", "--cached", "--stat"])
    say(f"staged:\n{stat}")

    msg = message or f"Cloud run results — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
    _run(["git", "commit", "-q", "-m", msg], check=True)

    url = f"https://{token}@github.com/{user}/{repo}.git"

    # Rebase onto the remote so a laptop-side commit does not reject the push.
    _run(["git", "fetch", "-q", url, branch], token=token)
    rc, out = _run(["git", "rebase", "-q", "FETCH_HEAD"], token=token)
    if rc != 0:
        _run(["git", "rebase", "--abort"])
        fallback = f"colab-results-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        say(f"rebase conflicted; pushing to branch '{fallback}' instead")
        _run(["git", "push", "-q", url, f"HEAD:refs/heads/{fallback}"], token=token, check=True)
        say(f"pushed -> {fallback} (open a PR to merge into {branch})")
        return True

    rc, out = _run(["git", "push", "-q", url, f"HEAD:refs/heads/{branch}"], token=token)
    if rc != 0:
        raise RuntimeError(f"push failed:\n{out}")
    say(f"pushed {len(existing)} path(s) -> github.com/{user}/{repo} ({branch})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--repo", default="ae496-ugp")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--paths", nargs="*", default=None)
    ap.add_argument("--message", default=None)
    args = ap.parse_args()
    ok = push(user=args.user, repo=args.repo, branch=args.branch,
              paths=args.paths, message=args.message)
    sys.exit(0 if ok else 0)


if __name__ == "__main__":
    main()
