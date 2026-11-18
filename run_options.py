"""Shared command-line paths and display options; no analysis logic."""

import argparse
from pathlib import Path


def parse_run_options(part_name: str, description: str) -> argparse.Namespace:
    """Resolve an explicit dataset path or discover one unambiguous location."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir", type=Path,
        help="Folder containing train.csv and the image paths listed in it.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "outputs",
        help="Output root; results are saved in a part-specific subfolder.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save figures without opening windows (also suitable for servers).",
    )
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
        if not (data_dir / "train.csv").is_file():
            parser.error(f"train.csv was not found in {data_dir}. Check --data-dir.")
    else:
        # Support data beside the scripts, beside their folder, or in the working directory.
        candidates = list(dict.fromkeys(path.resolve() for path in [script_dir, script_dir.parent, Path.cwd()]))
        matches = [path for path in candidates if (path / "train.csv").is_file()]
        if len(matches) != 1:
            reason = "No dataset was found" if not matches else "More than one dataset was found"
            parser.error(f'{reason}. Specify its location with --data-dir "path/to/data".')
        data_dir = matches[0]

    args.data_dir = data_dir
    args.output_dir = args.output_dir.expanduser().resolve() / part_name
    return args
