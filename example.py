#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from objects import *
### Check command-line arguments ###
if len(sys.argv) not in [2]:
    sys.exit("Usage: python example.py num_procs")

### Number of parallel processes ###
num_procs = int(sys.argv[1])

### Simulation parameters ###
species = ["Re", "Ox"]
diffusivities = [1E-9, 1E-9]
initial_concs = [1., 0.]
electro_rxns = [("Ox", "Re", 1E-2, 0.5, 1, 0.05),]
chem_rxns = []
x_d = 1E-5
A = 1E-4
V = 1E-6
T = 298
model = 0       # Type of potential
E_amp = 0.10
E_appl = 0.
freq = 1
floquet_tol = 1E-1

# Choose a value such that the waveform can be adequately described by guess_num_int steps
guess_num_int = int(1E2)

### Prepare potential ###
pot = Potential(model=model)
    
### Compute results ###
cell = AC_electrolytic_cell(species, diffusivities, initial_concs, electro_rxns, chem_rxns, x_d, A, V, T, pot, E_amp, E_appl, freq, floquet_tol=floquet_tol, guess_num_int=guess_num_int, num_procs=num_procs)
cell.solve_and_check()
cell.save("", "example")
cell.plot_summary("", "example")
cell.plot_fund_solns("", "example")

sys.stdout.flush()      # Flush all output from the process