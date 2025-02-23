#!/bin/bash

sourve .venv/bin/activate
pip install . --no-deps
python3 src/double_pendulum/transverse_dynamics/transverse_dynamics_test.py
