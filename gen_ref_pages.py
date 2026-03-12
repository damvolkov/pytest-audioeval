"""Auto-generate API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

SRC = Path("src")

for path in sorted(SRC.rglob("*.py")):
    module_path = path.relative_to(SRC)
    doc_path = path.relative_to(SRC).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.with_suffix("").parts)

    # Skip __main__, __init__, private modules
    if parts[-1].startswith("_"):
        continue

    identifier = ".".join(parts)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
