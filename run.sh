#!/bin/bash

sourve .venv/bin/activate
pip install . --no-deps
python3 src/transverse_dynamics/transverse_coordinates_sample.py
