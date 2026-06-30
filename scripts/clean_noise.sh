#!/usr/bin/env bash
# scripts/clean_noise.sh — purge the 3 classes of kernel-run noise.
#
# Matches the patterns in `.gitignore` (`*.log`, `RC=*`, `index.html`).
# Default: dry-run only — pass --force to actually wipe.
#
# Wraps `git clean -fdX -- <pathspec...>` so:
#   - pathspec scoping keeps the scope to the 3 noise classes (other
#     gitignored dirs like crates/cust_derive/wip/, guide/book/, .claude/,
#     /target, etc. stay intact);
#   - `-X` (--exclude-standard inverted) means "ignored-only", so this
#     script can never touch a tracked file even by accident.
#
# Exit codes:
#   0  clean (no matching files) OR successful wipe after --force
#   2  bad invocation / unknown flag

set -euo pipefail

DRY_RUN=1
QUIET=0

usage() {
  cat <<'EOF'
clean_noise.sh — purge kernel-run noise (3 classes: *.log, RC=*, index.html)

Usage:
  scripts/clean_noise.sh [--force] [--quiet]
  scripts/clean_noise.sh -h | --help

Options:
  -f, --force     Actually delete (default: dry-run only).
  -q, --quiet     Suppress per-file echo on delete (still prints a count
                  when running).
  -h, --help      Show this usage.

Default is dry-run to prevent accidental data loss; pass --force to
actually wipe. Pair this with `cargo run --release -p <kernel>`:

  scripts/clean_noise.sh                # see what each run left behind
  scripts/clean_noise.sh --force --quiet  # nuke before next run

Narrow scope by design: only the 3 noise classes are touched. Build
outputs (/target), book/, .claude/, .dejavue/ internals, etc. are
gitignored but OUTSIDE the 3 patterns and therefore preserved.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -f|--force) DRY_RUN=0; shift ;;
    -q|--quiet) QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)
      printf 'clean_noise.sh: unknown flag: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'clean_noise.sh: DRY-RUN — pass --force to actually delete.\n'
  exec git clean -fdnX -- '**/*.log' '**/RC=*' '**/index.html'
fi

# Actual delete path. Pathspec uses `**/…` so the match mirrors .gitignore's
# recursive scope (git pathspec default — `*` does not cross `/` — would
# silently miss any noise written into a subdirectory).
if [ "$QUIET" -eq 1 ]; then
  exec git clean -fqX -- '**/*.log' '**/RC=*' '**/index.html'
else
  exec git clean -fdX -- '**/*.log' '**/RC=*' '**/index.html'
fi
