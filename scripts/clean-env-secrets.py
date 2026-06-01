#!/usr/bin/env python3
"""Git clean filter for env files.

Reads env-file text from stdin and writes a redacted version to stdout.
This affects staged/committed content only when wired through Git's
filter.envsecrets.clean setting; it does not edit the working tree file.
"""

import re
import sys

SECRET_KEYS = {
    "LABEL_STUDIO_API_KEY",
    "LABEL_STUDIO_ACCESS_TOKEN",
    "BASIC_AUTH_PASS",
}

LINE_RE = re.compile(r"^(\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*)(.*?)(\s*)$")


def redact_line(line: str) -> str:
    match = LINE_RE.match(line.rstrip("\n"))
    if not match:
        return line

    prefix, key, value, suffix = match.groups()
    if key not in SECRET_KEYS or value == "":
        return line
    return f"{prefix}REDACTED{suffix}\n"


def main() -> int:
    for line in sys.stdin:
        sys.stdout.write(redact_line(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
