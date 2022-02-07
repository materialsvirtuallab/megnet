#!/usr/bin/env python
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
A master convenience script with many tools for vasp and structure analysis.
"""

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

from pymatgen.core import Structure
from tabulate import tabulate

from megnet.utils.models import MEGNetModel

DEFAULT_MODEL_PATH = Path(__file__).parent / ".." / ".." / "mvl_models" / "mp-2019.4.1"
DEFAULT_MODELS = [str(f) for f in DEFAULT_MODEL_PATH.glob("*.hdf5")]


def predict(args):
    """
    Handle view commands.

    :param args: Args from command.
    """
    headers = ["Filename"]
    output = []
    models = []
    prefix = ""
    for i, mn in enumerate(args.models):
        model = MEGNetModel.from_file(mn)
        models.append(model)
        if i == 0:
            prefix = mn
        else:
            sm = SequenceMatcher(None, prefix, mn)
            match = sm.find_longest_match(0, len(prefix), 0, len(mn))
            prefix = prefix[0 : match.size]
        headers.append(f"{mn} ({model.metadata.get('unit', '').strip('log10')}")
    headers = [h.lstrip(prefix) for h in headers]

    for fn in args.structures:
        structure = Structure.from_file(fn)
        row = [fn]
        for model in models:
            val = model.predict_structure(structure).ravel()
            if "log10" in str(model.metadata.get("unit", "")):
                val = 10**val
            row.append(val)
        output.append(row)
    print(tabulate(output, headers=headers))


def main():
    """
    Handle main.
    """
    parser = argparse.ArgumentParser(
        description="""
    meg is command-line interface to useful MEGNet tasks, e.g., prediction
    using a built model, etc. To see the options for the
    sub-commands, type "meg sub-command -h"."""
    )

    subparsers = parser.add_subparsers()

    parser_predict = subparsers.add_parser("predict", help="Predict property using MEGNET.")

    parser_predict.add_argument(
        "-s", "--structures", dest="structures", type=str, nargs="+", help="Structures to process"
    )
    parser_predict.add_argument(
        "-m", "--models", dest="models", type=str, nargs="+", default=DEFAULT_MODELS, help="Models to run."
    )
    parser_predict.set_defaults(func=predict)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
