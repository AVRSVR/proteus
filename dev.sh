#!/usr/bin/env bash
# Dev interpreter for Proteus.
#   miniconda base (3.13.9): working BLAS, Bio, pytest, openmm, pdbfixer.
#   NOTE: the design_env conda env has a broken numpy BLAS -- 'a @ b' hard-kills
#   the interpreter with no traceback. Do not develop against it.
export PROTEUS_PY="/c/Users/anant/miniconda3/python.exe"
exec "$PROTEUS_PY" "$@"
