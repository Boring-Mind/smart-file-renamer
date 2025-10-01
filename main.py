#!/usr/bin/env python3
"""
Smart music file renamer

Features
--------
- Learns an OLD filename template from your sample “old” names using RapidFuzz LCS.
- Builds a NEW template from a desired example/pattern that can mix:
    • inferred slots: {S1}, {S2}, ...
    • audio tags: {artist}, {album}, {title}, {track}, {disc}, {year}, {genre}
      (zero-padding like {track:02} supported)
- Dry-run, collision handling, keep/force extension, create subdirs.

Install:
    uv sync
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mutagen
from rapidfuzz.distance import LCSseq

# try:
#     from slugify import slugify as _slugify
#     HAVE_SLUG = True
# except Exception:
#     HAVE_SLUG = False


def debug(msg: str, enabled: bool):
    if enabled:
        print(f"[debug] {msg}")


def read_dir(dirpath: str, include_hidden: bool) -> List[str]:
    names = []
    for name in os.listdir(dirpath):
        if not include_hidden and name.startswith("."):
            continue
        if os.path.isfile(os.path.join(dirpath, name)):
            names.append(name)
    return sorted(names)


TemplatePiece = Tuple[str, str]  # (kind, value) where kind in {"lit","slot","tag"}


@dataclass
class Template:
    parts: List[TemplatePiece]

    def __str__(self):
        out = []
        for k, v in self.parts:
            if k == "lit":
                out.append(v)
            elif k == "slot":
                out.append(f"{{S{v}}}")
            else:  # tag
                out.append(f"{{{v}}}")
        return "".join(out)

    def slot_count(self) -> int:
        return max([int(v) for k, v in self.parts if k == "slot"] + [0])

    def parse(self, text: str) -> Optional[List[str]]:
        """
        Extract slot values from text according to this template.
        Tag parts are treated as literals during parsing (they don't extract values).
        Returns list of slot strings (1..slot_count), or None if not matching.
        """
        pattern = ["^"]
        slot_indices: List[int] = []
        for k, v in self.parts:
            if k == "lit":
                pattern.append(re.escape(v))
            elif k == "slot":
                pattern.append("(.*?)")
                slot_indices.append(int(v) - 1)
            elif k == "tag":
                # tags are not parsed from the filename; treat as literal placeholder text if present
                pattern.append(re.escape("{" + v + "}"))
        pattern.append("$")
        rx = re.compile("".join(pattern))
        m = rx.match(text)
        if not m:
            return None
        groups = list(m.groups())
        values = [None] * self.slot_count()
        gi = 0
        for k, v in self.parts:
            if k == "slot":
                values[int(v) - 1] = groups[gi]
                gi += 1
        if any(x is None for x in values):
            return None
        return values  # type: ignore

    def render(self, slot_values: List[str], tags: Dict[str, str]) -> str:
        out: List[str] = []
        for k, v in self.parts:
            if k == "lit":
                out.append(v)
            elif k == "slot":
                out.append(slot_values[int(v) - 1])
            else:  # tag
                out.append(tags.get(v, ""))
        return "".join(out)


def strip_ext(name: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(name)
    return base, ext


def lcs_multi(strings: List[str]) -> str:
    """
    Build a multi-string LCS by reducing pairwise using RapidFuzz opcodes.
    """
    if not strings:
        return ""
    base = strings[0]
    for s in strings[1:]:
        # Collect contiguous equal blocks (the LCS sequence)
        ops = LCSseq.opcodes(base, s)
        pieces = [base[i1:i2] for (tag, i1, i2, j1, j2) in ops if tag == "equal"]
        base = "".join(pieces)
    return base


def build_old_template(example_olds: List[str], verbose: bool = False) -> Template:
    bases = [strip_ext(x)[0] for x in example_olds]
    lcs = lcs_multi(bases)
    debug(f"LCS across olds: {repr(lcs)}", verbose)

    base0 = bases[0]
    parts: List[TemplatePiece] = []

    # Two-pointer scan to find in-order anchors of lcs in base0
    i = j = 0
    anchors: List[Tuple[int, int]] = []  # (start, end) in base0
    while i < len(base0) and j < len(lcs):
        if base0[i] == lcs[j]:
            start = i
            while i < len(base0) and j < len(lcs) and base0[i] == lcs[j]:
                i += 1
                j += 1
            anchors.append((start, i))
        else:
            i += 1

    # Merge adjacent anchors
    merged: List[Tuple[int, int]] = []
    for st, en in anchors:
        if not merged:
            merged.append((st, en))
        else:
            pst, pen = merged[-1]
            if pen == st:
                merged[-1] = (pst, en)
            else:
                merged.append((st, en))

    slot_idx = 1
    cursor = 0
    for st, en in merged:
        if st > cursor:
            parts.append(("slot", str(slot_idx)))
            slot_idx += 1
        lit = base0[st:en]
        if lit:
            parts.append(("lit", lit))
        cursor = en
    if cursor < len(base0):
        parts.append(("slot", str(slot_idx)))

    # Extension from the first example as literal (keeps matches scoped to ext)
    ext0 = strip_ext(example_olds[0])[1]
    if ext0:
        parts.append(("lit", ext0))

    tpl = Template(parts)
    debug(f"OLD template: {tpl}", verbose)
    return tpl


SUPPORTED_TAGS = {
    "artist",
    "album",
    "title",
    "track",
    "disc",
    "year",
    "genre",
}


def parse_desired_to_template(
    desired: str, slot_values_from_old0: List[str], verbose: bool = False
) -> Template:
    """
    Build a NEW template by:
      1) Recognizing explicit tag placeholders like {artist}, {track:02}
      2) Replacing occurrences of slot values (from old0) with {S#} in order
      3) Everything else is literal

    Tag parts are stored as ('tag', 'name[:format]') to preserve requested formatting.
    """
    parts: List[TemplatePiece] = []

    # Split desired into chunks by tag placeholders
    tag_rx = re.compile(r"\{([a-zA-Z]+)(?::([^{}]+))?\}")

    pos = 0
    tokens: List[Tuple[str, str]] = []  # (type, text) type in {'lit','tag'}
    for m in tag_rx.finditer(desired):
        if m.start() > pos:
            tokens.append(("lit", desired[pos : m.start()]))
        name = m.group(1).lower()
        fmt = m.group(2) or ""
        tok = f"{name}:{fmt}" if fmt else name
        if name in SUPPORTED_TAGS:
            tokens.append(("tag", tok))
        else:
            tokens.append(("lit", m.group(0)))
        pos = m.end()
    if pos < len(desired):
        tokens.append(("lit", desired[pos:]))

    # Within literal chunks, replace slot values in order with slot tokens
    for t, txt in tokens:
        if t == "tag":
            parts.append(("tag", txt))
        else:
            lit = txt
            cur = 0
            for idx, val in enumerate(slot_values_from_old0, start=1):
                if not val:
                    continue
                k = lit.find(val, cur)
                if k == -1:
                    continue
                if k > cur:
                    parts.append(("lit", lit[cur:k]))
                parts.append(("slot", str(idx)))
                cur = k + len(val)
            if cur < len(lit):
                parts.append(("lit", lit[cur:]))

    if not parts:
        parts = [("lit", desired)]

    if verbose:
        debug(f"NEW template parsed: {Template(parts)}", True)
    return Template(parts)


def read_tags(filepath: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    try:
        audio = mutagen.File(filepath, easy=True)
        if audio is None:
            return tags

        def first(key: str) -> str:
            v = audio.get(key)
            if isinstance(v, list):
                return str(v[0]) if v else ""
            return str(v) if v is not None else ""

        tags["artist"] = first("artist")
        tags["album"] = first("album")
        tags["title"] = first("title")
        tags["genre"] = first("genre")

        def first_num(s: str) -> str:
            if not s:
                return ""
            s = str(s)
            m = re.match(r"\s*(\d+)", s)
            return m.group(1) if m else ""

        tags["track"] = first_num(first("tracknumber") or first("track"))
        tags["disc"] = first_num(first("discnumber") or first("disc"))

        date = first("date") or first("year")
        m = re.search(r"(\d{4})", date) if date else None
        tags["year"] = m.group(1) if m else ""

    except Exception:
        return tags
    return tags


def apply_format(value: str, fmt: str) -> str:
    if not fmt:
        return value
    # Only simple zero-padding formatting like 02, 03 supported
    m = re.fullmatch(r"0?(\d+)", fmt)
    if m and value.isdigit():
        width = int(m.group(1))
        return value.zfill(width)
    return value


def ensure_unique(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = f"{root} ({n}){ext}"
        if not os.path.exists(cand):
            return cand
        n += 1


def apply_case(s: str, mode: str) -> str:
    if mode == "lower":
        return s.lower()
    if mode == "upper":
        return s.upper()
    if mode == "title":

        def cap(match: "re.Match[str]") -> str:
            w = match.group(0)
            return w if w.isupper() else (w[:1].upper() + w[1:])

        return re.sub(r"[A-Za-z][A-Za-z0-9]*", cap, s)
    if mode in ("slug", "underscore"):
        separator = "-" if mode == "slug" else "_"
        return s.replace(" ", separator)
        # if HAVE_SLUG:
        #     return _slugify(s, lowercase=False)
        # t = re.sub(r"[^\w\-\s]+", '', s)
        # t = re.sub(r"\s+", '-', t).strip('-')
        # return t
    return s


def render_with_tags(
    tpl: Template, slot_values: List[str], tags: Dict[str, str], case_mode: str
) -> str:
    rendered_parts: List[str] = []
    for k, v in tpl.parts:
        if k == "lit":
            rendered_parts.append(v)
        elif k == "slot":
            rendered_parts.append(apply_case(slot_values[int(v) - 1], case_mode))
        else:  # tag
            if ":" in v:
                name, fmt = v.split(":", 1)
            else:
                name, fmt = v, ""
            raw = tags.get(name, "")
            raw = apply_format(raw, fmt)
            rendered_parts.append(apply_case(raw, case_mode))
    return "".join(rendered_parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Infer a renaming rule from examples and apply to a folder (tag-aware)."
    )
    ap.add_argument(
        "--dir", required=True, help="Target directory containing the files to rename"
    )
    ap.add_argument(
        "--olds",
        nargs="+",
        required=True,
        help="One or more CURRENT filenames (relative to --dir). The FIRST will be paired with --desired",
    )
    ap.add_argument(
        "--desired",
        required=True,
        help="Desired name/pattern. May include {artist},{album},{title},{track},{disc},{year},{genre} and {S#}. Can include directories.",
    )
    ap.add_argument(
        "--old-pattern",
        help="Provide a custom pattern if you see that auto-generated pattern is not precise.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, do not rename"
    )
    ap.add_argument(
        "--conflict",
        choices=["skip", "overwrite", "unique"],
        default="unique",
        help="Collision handling when target exists",
    )
    ap.add_argument(
        "--ext-mode",
        choices=["keep", "desired"],
        default="keep",
        help="Keep original extensions or use the desired example's extension for all",
    )
    ap.add_argument(
        "--case-slots",
        choices=["asis", "lower", "upper", "title", "slug", "underscore"],
        default="asis",
        help="Apply casing/normalization to slot values and tag values",
    )
    ap.add_argument(
        "--mkdirs",
        dest="mkdirs",
        action="store_true",
        default=True,
        help="Create directories in desired pattern if they do not exist (default)",
    )
    ap.add_argument("--no-mkdirs", dest="mkdirs", action="store_false")
    ap.add_argument(
        "--require-tags",
        action="store_true",
        help="Skip files that are missing any referenced tag placeholders",
    )
    ap.add_argument(
        "--include-hidden", action="store_true", help="Include dotfiles in scans"
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose debug logs")

    args = ap.parse_args(argv)

    dirpath = os.path.abspath(args.dir)
    if not os.path.isdir(dirpath):
        print(f"! Not a directory: {dirpath}", file=sys.stderr)
        return 2

    olds = args.olds
    for name in olds:
        if not os.path.exists(os.path.join(dirpath, name)):
            print(f"! Example old not found: {name}", file=sys.stderr)
            return 2

    # ToDo: add here custom template if provided, instead of learning it on the fly.
    # ToDo: add possibility to actually rename all files in this directory (not recursively)
    #  Also, you should be able to provide extension of files that have to be renamed.

    # Learn OLD template from examples
    old_tpl = build_old_template(olds, verbose=args.verbose)

    # Parse the first old to get slot values
    first_old = olds[0]
    slot_vals_old0 = old_tpl.parse(first_old)
    if slot_vals_old0 is None:
        print(
            "! Could not parse the first example old with the learned OLD template. Give more/cleaner examples.",
            file=sys.stderr,
        )
        return 3

    # Build NEW template using desired pattern (with {tags} and/or slots)
    desired = args.desired
    desired_ext = os.path.splitext(desired)[1]
    new_tpl = parse_desired_to_template(desired, slot_vals_old0, verbose=args.verbose)

    # Scan target directory for candidates matching OLD template
    all_files = read_dir(dirpath, args.include_hidden)
    candidates: List[str] = []
    for fn in all_files:
        if old_tpl.parse(fn) is not None:
            candidates.append(fn)

    if not candidates:
        print(
            "No files in the folder matched the learned OLD template. Refine your examples."
        )
        return 0

    print("Learned:")
    print(f"  OLD: {old_tpl}")
    print(f"  NEW: {new_tpl}")
    print("")

    # Prepare renames
    plan: List[Tuple[str, str, str]] = []  # (src_path, dst_path, action)
    missing_tag_files = 0
    for src in candidates:
        src_path = os.path.join(dirpath, src)
        slot_vals = old_tpl.parse(src) or []

        # Read tags
        tags = read_tags(src_path)

        # Enforce required tags if requested
        if args.require_tags:
            needed = []
            for k, v in new_tpl.parts:
                if k == "tag":
                    name = v.split(":", 1)[0]
                    needed.append(name)
            if any(not tags.get(n) for n in needed):
                missing_tag_files += 1
                plan.append((src_path, src_path, "skip"))
                continue

        # Render target name/path
        dst_rel = render_with_tags(new_tpl, slot_vals, tags, args.case_slots)

        # Extension policy
        if args.ext_mode == "keep":
            src_ext = os.path.splitext(src)[1]
            if src_ext:
                base, _ = os.path.splitext(dst_rel)
                dst_rel = base + src_ext
        else:  # desired
            if desired_ext:
                base, _ = os.path.splitext(dst_rel)
                dst_rel = base + desired_ext

        dst_path = os.path.join(dirpath, dst_rel)

        # Ensure directories exist
        if args.mkdirs:
            os.makedirs(os.path.dirname(dst_path) or dirpath, exist_ok=True)

        if src_path == dst_path:
            plan.append((src_path, dst_path, "noop"))
        else:
            plan.append((src_path, dst_path, "rename"))

    # Resolve conflicts
    resolved: List[Tuple[str, str, str]] = []
    for src, dst, action in plan:
        if action != "rename":
            resolved.append((src, dst, action))
            continue
        if os.path.exists(dst):
            if args.conflict == "skip":
                resolved.append((src, dst, "skip"))
            elif args.conflict == "overwrite":
                resolved.append((src, dst, "overwrite"))
            else:  # unique
                resolved.append((src, ensure_unique(dst), "unique"))
        else:
            resolved.append((src, dst, "rename"))

    # Report
    print("Planned actions:")
    sym = {
        "noop": " = ",
        "rename": " → ",
        "skip": " × ",
        "overwrite": " ↻ ",
        "unique": " ⊕ ",
    }
    for src, dst, act in resolved:
        print(
            f"  {os.path.relpath(src, dirpath)}{sym[act]}{os.path.relpath(dst, dirpath)}"
        )

    if args.dry_run:
        if missing_tag_files:
            print(
                f"\nNote: {missing_tag_files} file(s) skipped due to missing required tags."
            )
        print("\n(dry-run) No changes were made.")
        return 0

    # Perform
    ran = 0
    skipped = 0
    overwritten = 0
    uniqued = 0
    for src, dst, act in resolved:
        if act == "noop":
            continue
        if act == "skip":
            skipped += 1
            continue
        if act in ("rename", "overwrite", "unique"):
            os.rename(src, dst)
            if act == "unique":
                uniqued += 1
            elif act == "overwrite":
                overwritten += 1
            else:
                ran += 1

    print(
        f"Done. Renamed: {ran}, Overwritten: {overwritten}, Unique-renamed: {uniqued}, Skipped: {skipped}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
