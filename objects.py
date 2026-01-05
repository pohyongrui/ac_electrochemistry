# -*- coding: utf-8 -*-

import os
import sys
import platform
import numpy as np
import dill
import inspect
import warnings
import seaborn as sns
import pandas as pd
import itertools
from floquet import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
### Choose alternative font on supercomputer ###
if platform.system() == "Linux":
    font = "Nimbus Sans"
else:
    font = "Arial"
plt.rcParams.update({
    "text.usetex": False,
    "font.family": font,
    "mathtext.fontset": "custom",
    "mathtext.rm": font,
    "mathtext.it": f"{font}:italic",
    "axes.formatter.useoffset": False
})

class Potential:
    """
    Prepares the oscillating function of the applied alternating potential.
    Choose from one of the default potentials or define a custom potential.
    For custom potential, (1) potential must be function of frequency * time with period 1, and (2) value at t = 0 must be zero.
    Note that the DC potential is approximated by the square of the positive half cycle of the squdel function (adjusted to have 
    a tau period of 1).
    
    Attributes:
        model: Type of potential. 0 = Sine wave; 1 = Square wave; 3 = DC potential; 6 = Shifted square wave; 
               -1 = Custom potential. Defaults to 0.
        
        **kwargs
        kappa: (Only for type = 1, 3, or 6) kappa for either of the following functions:
                - squdel function, which approches the square wave as kappa approaches zero
                - shifted squdel function, which approaches the shifted square wave as kappa approaches zero
                Defaults to 0.01.
        epsilon: Downward shift in the squdel function for the shifted squdel function (only for type = 6)
        pot: Function describing custom potential (only for type = -1)
        pot_name: Name of custom potential (only for type = -1)
    """
    
    ### Prepare list of default oscillating potentials ###
    ### Note: tau = frequency * time ###
    @staticmethod
    def sine_wave(tau):
        return np.sin(2 * np.pi * tau)

    @staticmethod
    def squdel_function(tau,kappa):
        return np.sin(2 * np.pi * tau) / np.sqrt(np.sin(2 * np.pi * tau)**2 + kappa**2 * np.cos(2 * np.pi * tau)**2)
    
    @staticmethod
    def DC_potential(tau,kappa):
        return np.sin(2 * np.pi * tau / 2)**2 / (np.sin(2 * np.pi * tau / 2)**2 + kappa**2 * np.cos(2 * np.pi * tau / 2)**2)
    
    @staticmethod
    def shifted_squdel_function(tau,kappa,sigma):
        return Potential.squdel_function((tau + sigma),kappa) - Potential.squdel_function(sigma,kappa)
        
    pot_list = [sine_wave.__func__,
                squdel_function.__func__,
                None,
                DC_potential.__func__,
                None,
                None,
                shifted_squdel_function.__func__]
    pot_names = ["Sine wave","Square wave",None,"DC potential",None,None,"Shifted square wave"]
    
    def __init__(self,
                 model = 0,
                 **kwargs):
        self.model = model
        
        if self.model == -1:
            self.pot = kwargs["pot"]
            self.pot_name = kwargs["pot_name"]
            
        else:
            if self.model in (1,3,6):
                try:
                    self.kappa = kwargs["kappa"]
                except KeyError:
                    self.kappa = 1E-2
                if self.model == 6:
                    self.epsilon = kwargs["epsilon"]
                    # Find sigma for shifted squdel function
                    self.sigma = optimize.root(lambda sigma: Potential.squdel_function(sigma,self.kappa) - self.epsilon, x0=[0,]).x[0]
                    self.pot = lambda tau: self.pot_list[self.model](tau,self.kappa,self.sigma)
                else:
                    self.pot = lambda tau: self.pot_list[self.model](tau,self.kappa)
            else:
                self.pot = lambda tau: self.pot_list[self.model](tau)
            self.pot_name = self.pot_names[self.model]
            
    def __str__(self):
        return self.pot_name

class AC_electrolytic_cell:
    """
    Simulates a general first-order AC electrolysis reaction using Floquet theory.
    NOTE: All input variables MUST be entered in SI units. All output variables will be printed in SI units.
    
    Attributes:
        species: List containing the name of all species as strings. No duplicates are allowed.
        diffusivities: List containing the diffusion constants of all species, presented in the same order as "species"
        initial_concs: List containing the initial concentrations of all species, presented in the same order as "species"
        electro_rxns: List containing the electrochemical reactions, each presented as a tuple in the following order:
            oxidant: Name of oxidant species. Must be a string that matches an element of "species".
            reductant: Name of reductant species. Must be a string that matches an element of "species".
            k_O_R: Standard electrochemical rate constant
            alpha_O_R: Butler-Volmer transfer coefficient
            n_O_R: Number of electrons transferred per redox reaction
            E_O_R: Standard reduction potential relative to a fixed reference
        chem_rxns: List containing the chemical reactions, each presented as a tuple in the following order:
            A: Name of reactant species. Must be a string that matches an element of "species".
            B: Name of product species. Must be a string that matches an element of "species".
            k_A_B: Forward rate constant
            k_B_A: Backward rate constant
        x_d: Diffusion layer thickness
        A: Electrode surface area
        V: Electrolytic cell volume
        T: Temperature
        pot: Potential object containing oscillating function of applied potential
        E_amp: Amplitude of potential oscillations
        E_appl: Average applied potential relative to a fixed reference
        freq: Frequency of AC oscillations
        
        **kwargs
        min_num_int: Minimum number of intervals to divide the period into during numerical integration.
                        If this value is too small, Floquet_solver will keep increasing it in multiples of 10 until a suitable value is found.
                        However, only min_num_int timepoints will be stored.
                        Defaults to 100.
        guess_num_int: Guess number of intervals to divide the period into during numerical integration.
                        Must be an integer multiple of min_num_int.
                        Defaults to min_num_int.
        floquet_tol: Tolerance for Floquet_solver. Defaults to 1E-1.
        num_procs: Number of parallel processes to perform. Defaults to 1.
        method: Method for Floquet_solver.
                    1 = Expanding the exponential (old algorithm).
                    2 = Not expanding the exponential (new algorithm).
                    Defaults to 2.
        num_tol: Threshold below which any numerical value is considered zero. Defaults to 1E-3.
        oxidation_only: Boolean for enforcing only oxidations (i.e. no reductions). Defaults to False.
        reduction_only: Boolean for enforcing only reductions (i.e. no oxidations). Defaults to False.
    """
    
    @staticmethod
    def test_if_equal(a, b, tol):
        """
        Tests if a and b are nearly equal.
        
        Attributes:
            a, b: Values to be compared
            tol: Tolerance for error
            
        Returns:
            Boolean describing test result.
        """
        
        a_abs, b_abs, diff = abs(a), abs(b), abs(a - b)
        if a == b: 
            # Shortcut, handles infinities
            return True
        elif a == 0 or b == 0 or a_abs + b_abs < tol:
            # a or b is zero or both are extremely close to it, so relative error is less meaningful here
            return diff < tol
        else:
            # Use relative error
            return diff / (a_abs + b_abs) < tol
        
    @staticmethod
    def test_if_vectors_equal(A, B, tol):
        """
        Tests if vectors A and B are nearly equal using test_if_equal function.
        
        Attributes:
            A, B: Numpy vectors to be compared
            tol: Tolerance for error
            
        Returns:
            Boolean describing test result.
        """
        
        for (a,b) in np.column_stack((A,B)):
            if not AC_electrolytic_cell.test_if_equal(a, b, tol):
                return False
        return True
    
    def __init__(self,
                 species: list,
                 diffusivities: list,
                 initial_concs: list,
                 electro_rxns: list,
                 chem_rxns: list,
                 x_d: float,
                 A: float,
                 V: float,
                 T: float,
                 pot: Potential,
                 E_amp: float,
                 E_appl: float,
                 freq: float,
                 **kwargs):
    
        self.species = species
        self.diffusivities = diffusivities
        self.initial_concs = initial_concs
        self.electro_rxns = electro_rxns
        self.chem_rxns = chem_rxns
        self.x_d = x_d
        self.A = A
        self.V = V
        self.T = T
        self.pot = pot      
        self.E_amp = E_amp
        self.E_appl = E_appl
        self.freq = freq
        try:
            self.min_num_int = kwargs["min_num_int"]
        except KeyError:
            self.min_num_int = 100
        try:
            self.guess_num_int = kwargs["guess_num_int"]
        except KeyError:
            self.guess_num_int = self.min_num_int
        try:
            self.floquet_tol = kwargs["floquet_tol"]
        except (KeyError, NameError):
            self.floquet_tol = 1E-1
        try:
            self.num_procs = kwargs["num_procs"]
        except KeyError:
            self.num_procs = 1
        try:
            self.method = kwargs["method"]
        except KeyError:
            self.method = 2
        try:
            self.num_tol = kwargs["num_tol"]
        except (KeyError, NameError):
            self.num_tol = 1E-3
        try:
            self.oxidation_only = kwargs["oxidation_only"]
        except (KeyError, NameError):
            self.oxidation_only = False
        try:
            self.reduction_only = kwargs["reduction_only"]
        except (KeyError, NameError):
            self.reduction_only = False
    
        # Check that all inputs are valid
        if len(self.species) != len(set(self.species)):
            raise TypeError(f"Species should not have any duplicate values.")
        if not all(isinstance(specie, str) for specie in self.species):
            raise TypeError(f"All species should be entered as strings.")
        if len(self.species) != len(self.diffusivities):
            raise TypeError(f"The number of species must match the number of diffusion constants. Number of species: {len(self.species)}. Number of diffusion constants: {len(self.diffusivities)}.")
        if not all(isinstance(diffusivity, float) for diffusivity in self.diffusivities):
            raise TypeError(f"All diffusion constants should be entered as floats.")
        if len(self.species) != len(self.initial_concs):
            raise TypeError(f"The number of initial concentrations must match the number of species. Number of species: {len(self.species)}. Number of initial concentrations: {len(self.initial_concs)}.")
        if not all(isinstance(initial_conc, float) for initial_conc in self.initial_concs):
            raise TypeError(f"All initial concentrations should be entered as floats.")
        if not all(isinstance(electro_rxn, tuple) for electro_rxn in self.electro_rxns):
            raise TypeError(f"All electrochemical reactions should be entered as tuples.")
        for oxidant, reductant, k_O_R, alpha_O_R, n_O_R, E_O_R in self.electro_rxns:
            if oxidant not in self.species:
                raise TypeError(f"Oxidant {oxidant} not defined in species.")
            if reductant not in self.species:
                raise TypeError(f"Reductant {reductant} not defined in species.")
            if type(k_O_R) != float:
                raise TypeError(f"k_O_R for {oxidant} <--> {reductant} redox should be entered as a float. Type of k_O_R: {type(k_O_R)}.")
            if type(alpha_O_R) != float:
                raise TypeError(f"alpha_O_R for {oxidant} <--> {reductant} redox should be entered as a float. Type of alpha_O_R: {type(alpha_O_R)}.")
            if type(n_O_R) != float and type(n_O_R) != int:
                raise TypeError(f"n_O_R for {oxidant} <--> {reductant} redox should be entered as a float or integer. Type of n_O_R: {type(n_O_R)}.")
            if type(E_O_R) != float:
                raise TypeError(f"E_O_R for {oxidant} <--> {reductant} redox should be entered as a float. Type of E_O_R: {type(E_O_R)}.")
        if not all(isinstance(chem_rxn, tuple) for chem_rxn in self.chem_rxns):
            raise TypeError(f"All chemical reactions should be entered as tuples.")
        for A, B, k_A_B, k_B_A in self.chem_rxns:
            if A not in self.species:
                raise TypeError(f"Reactant {A} not defined in species.")
            if B not in self.species:
                raise TypeError(f"Product {B} not defined in species.")
            if type(k_A_B) != float:
                raise TypeError(f"k_A_B for {A} <--> {B} reaction should be entered as a float. Type of k_A_B: {type(k_A_B)}.")
            if type(k_B_A) != float:
                raise TypeError(f"k_B_A for {A} <--> {B} reaction should be entered as a float. Type of k_B_A: {type(k_B_A)}.")
        if type(self.min_num_int) != int:
            raise TypeError(f"min_num_int should be an integer. Type of min_num_int: {type(self.min_num_int)}.")
        if type(self.guess_num_int) != int:
            raise TypeError(f"guess_num_int should be an integer. Type of guess_num_int: {type(self.guess_num_int)}.")
        if self.guess_num_int % self.min_num_int:
            raise TypeError(f"guess_num_int must be divisible by min_num_int. guess_num_int: {self.guess_num_int}. min_num_int: {self.min_num_int}")  
        if type(self.floquet_tol) != float:
            raise TypeError(f"floquet_tol should be a float. Type of floquet_tol: {type(self.floquet_tol)}.")  
        if type(self.num_procs) != int:
            raise TypeError(f"num_procs should be a float. Type of num_procs: {type(self.num_procs)}.")
        if type(self.method) != int:
            raise TypeError(f"method should be an integer. Type of method: {type(self.method)}.")
        if type(self.num_tol) != float:
            raise TypeError(f"num_tol should be a float. Type of num_tol: {type(self.num_tol)}.")
        if type(self.oxidation_only) != bool:
            raise TypeError(f"oxidation_only should be a boolean. Type of oxidation_only: {type(self.oxidation_only)}.")
        if type(self.reduction_only) != bool:
            raise TypeError(f"reduction_only should be a boolean. Type of reduction_only: {type(self.reduction_only)}.")

        # Compute some universal parameters
        self.num_species = len(self.species)
        self.F = 9.64853321233100184E4  # Faraday constant
        self.R = 8.31446261815324   # Gas constant
        self.f = self.F / (self.R * self.T)

        # Convert the potential function from dimensionless time into real time, scaled by the amplitude and shifted by the average value
        self.E = lambda t: self.E_amp * self.pot.pot(self.freq * t) + self.E_appl        

        # Prepare the initial concentration vector
        self.c_0s = np.array([self.initial_concs[s // 2] for s in range(self.num_species * 2)])
        
        # Store all input parameters for printing
        self.properties = inspect.getargvalues(inspect.currentframe())[3]   # Stores all arguments of __init__ as a dictionary
        

    #########################################
    ### Functions that perform simulation ###
    #########################################

    def K(self, t):
        """
        Function generating the 2Nx2N rate constant matrix that is periodic in time.
        
        Attributes:
            t: Time
            
        Returns:
            K: A numpy array containing K at time t.
        """
        
        # Find the potential at time t
        E = self.E(t)
        
        # Prepare the three rate constant matrices
        K_D, K_E, K_C = np.zeros((self.num_species * 2, self.num_species * 2)), np.zeros((self.num_species * 2, self.num_species * 2)), np.zeros((self.num_species * 2, self.num_species * 2))
        
        # Populate the diffusion matrix
        for s, D_s in enumerate(self.diffusivities):
            K_D[2*s,2*s+1] += D_s / self.x_d**2
            K_D[2*s,2*s] += -D_s / self.x_d**2
            K_D[2*s+1,2*s+1] += - self.A * D_s / ((self.V - self.A * self.x_d) * self.x_d)
            K_D[2*s+1,2*s] += self.A * D_s / ((self.V - self.A * self.x_d) * self.x_d)
            
        # Populate the electron transfer matrix
        for oxidant, reductant, k_O_R, alpha_O_R, n_O_R, E_O_R in self.electro_rxns:
            # Index the chemical species
            oxidant_s, reductant_s = self.species.index(oxidant), self.species.index(reductant)
            
            if not self.oxidation_only:
                K_E[2*oxidant_s,2*oxidant_s] += -k_O_R / self.x_d * np.exp(-alpha_O_R * n_O_R * self.f * (E - E_O_R))
                K_E[2*reductant_s,2*oxidant_s] += k_O_R / self.x_d * np.exp(-alpha_O_R * n_O_R * self.f * (E - E_O_R))
                
            if not self.reduction_only:
                K_E[2*oxidant_s,2*reductant_s] += k_O_R / self.x_d * np.exp((1 - alpha_O_R) * n_O_R * self.f * (E - E_O_R))
                K_E[2*reductant_s,2*reductant_s] += -k_O_R / self.x_d * np.exp((1 - alpha_O_R) * n_O_R * self.f * (E - E_O_R))
        
        # Populate the chemical reaction matrix
        for A, B, k_A_B, k_B_A in self.chem_rxns:
            # Index the chemical species
            A_s, B_s = self.species.index(A), self.species.index(B)
            
            # Surface reactions
            K_C[2*A_s,2*A_s] += -k_A_B
            K_C[2*A_s,2*B_s] += k_B_A
            K_C[2*B_s,2*A_s] += k_A_B
            K_C[2*B_s,2*B_s] += -k_B_A
            
            # Bulk reactions
            K_C[2*A_s+1,2*A_s+1] += -k_A_B
            K_C[2*A_s+1,2*B_s+1] += k_B_A
            K_C[2*B_s+1,2*A_s+1] += k_A_B
            K_C[2*B_s+1,2*B_s+1] += -k_B_A
            
        K = K_D + K_E + K_C
        
        return K
        
    def get_c_ave(self, cs):
        """
        Finds the average concentration for some c-vector.
        
        Attributes:
            cs: c-vector
            
        Returns:
            c_ave: Average concentration for this c-vector
        """
        c_ave = 0
        for s in range(self.num_species):
            c_ave += self.A * self.x_d / self.V * cs[2*s] + (self.V - self.A * self.x_d) / self.V * cs[2*s+1]
        
        return c_ave
    
    def solve_and_check(self):
        """
        Solves the kinetic equations using Floquet theory.
        Checks that the characteristic exponents are all real-valued and negative except one that is zero.
        Checks that conservation of mass is obeyed in the steady-state outcome.
        
        Solutions are stored as such: 
            fund_solns: Fundamental solutions. First index labels the time, next two indices label the solution matrices.
            char_expons: Characteristic exponents
            coeffs: Expansion coefficients
            t_array: Discretised time array
            c_ss: Steady-state solution
            ss_timescale: Timescale towards steady state
        
        Attributes:
            None
        """
        
        # Solve the differential equations using Floquet theory
        print(f"\n=== NOW ENTERING THE FLOQUET SOLVER MODULE ===\n")
        self.solver = Floquet_solver(self.K, self.freq, self.c_0s, min_num_int=self.min_num_int, guess_num_int=self.guess_num_int, tol=self.floquet_tol, num_procs=self.num_procs, method=self.method)
        self.solver.solve()
        
        # Extract the solutions
        self.fund_solns, self.char_expons, self.coeffs, self.t_array = self.solver.fund_solns, self.solver.char_expons, self.solver.coeffs, self.solver.t_array
        print(f"\n=== NOW LEAVING THE FLOQUET SOLVER MODULE ===\n")
        
        
        ####################################################
        ### Check that the solutions are physically sound ##
        ####################################################
        
        # Check if all characteristic exponents have negative or zero real components, with at least one zero present
        # Note that Floquet_solver already sorted the solutions by the real parts of the characteristic exponents in ascending order
        char_expons_max, char_expons_next_max = self.char_expons[-1], self.char_expons[-2]
        if np.abs(char_expons_max.real / self.freq) > self.num_tol or char_expons_next_max.real / self.freq > self.num_tol:
            raise Exception(f"The one-period time propagator of Floquet theory was not solved correctly. Two largest characteristic exponents = {char_expons_max}, {char_expons_next_max}.")        
                
        
        ##########################################################################
        ### Extact the steady-state outcome and timescale towards steady state ###
        ##########################################################################
        
        # In this code, we will extract the steady-state outcome by first assuming the fundamental solutions and coefficients to be complex-valued 
        # before taking the real component of their product. This is done in case the coefficients (which in principle should be real-valued) 
        # have large imaginary components. The real-valued initial condition should ensure that at t=0, the steady-state outcome is real-valued.
        self.c_ss = self.fund_solns[0,-1] * self.coeffs[-1]
        self.c_ss = self.c_ss.real
        self.ss_timescale = -1 / self.char_expons[-2].real
        
        # Check that mass is conserved
        c_ave_0, c_ave_ss = self.get_c_ave(self.c_0s), self.get_c_ave(self.c_ss)
        if not AC_electrolytic_cell.test_if_vectors_equal(c_ave_ss, c_ave_0, self.num_tol):
            raise Exception(f"Conservation of mass is not obeyed in the steady-state outcome. Computed average concentration = {c_ave_ss}. Ideal average concentration = {c_ave_0}.")


    ############################################
    ### Functions that save and load results ###
    ############################################

    def save(self, subdir, filename):
        """
        Saves results in human-readable format (.txt).
        Also dills itself (.pkl).
        Note that the fundamental solutions are truncated in the human-readable output to only the t=0 solution.
        
        Attributes:
            subdir: Subdirectory to which the results should be written
            filename: Filename to which the results should be written
        """
        
        # Prepare filenames
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        # Write key results in human-readable format
        with open(filename_full + ".txt", "w") as file:
            file.write("=== INPUT PARAMETERS ===\n")
            for key, value in self.properties.items():
                if key == "self":
                    continue
                file.write(f"\n{key}:\n")
                file.write(f"{value}\n")
            file.write("\nIf no kwargs are present, then the default values were used in the simulation.\n")
            
            file.write("\n\n\n")
            
            file.write("\n=== RESULTS ===\n")
            file.write("\nFundamental solutions at t=0:\n")
            np.savetxt(file, self.fund_solns[0])
            file.write("\nCharacteristic exponents:\n")
            np.savetxt(file, self.char_expons)
            file.write("\nExpansion coefficients:\n")
            np.savetxt(file, self.coeffs)
            file.write("\nSteady-state solution:\n")
            np.savetxt(file, self.c_ss)
            file.write(f"\nTimescale towards steady state = {self.ss_timescale} s\n")
            
        # Dill itself
        with open(filename_full + ".pkl", "wb") as file:
            dill.dump(self, file)
        
        print(f"The following files were saved: ")
        print(f"\t {filename_full}.txt")
        print(f"\t {filename_full}.pkl")
        print("\n")
    
    @staticmethod
    def load(subdir, filename):
        """
        Loads results from the save file of dill format (.pkl).
        
        Usage: cell = AC_electrolytic_cell.load(subdir, filename)
        
        Attributes:
            subdir: Subdirectory from which the results should be loaded
            filename: Filename from which the results should be loaded
        """
        
        # Prepare filenames
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        with open(filename_full + ".pkl", "rb") as file:
            print(f"Loading the following file: ")
            print(f"\t {filename_full}.pkl")
            print("\n")
            return dill.load(file)


    #################################################################
    ### Functions that compute additional properties from results ###
    #################################################################

    def cs(self, t_j):
        """
        Finds the c-vector for discrete time t_j.
        
        Attributes:
            t_j: Discrete time
        """
        
        # Find the equivalent t_j in the first time period
        adjusted_t_j = t_j - int(t_j * self.freq) / self.freq
        
        # Check if t_j is a valid discrete time within some error
        try: 
            truth_array = [AC_electrolytic_cell.test_if_equal(t, adjusted_t_j, self.num_tol) for t in self.t_array]
            j = truth_array.index(True)
        except ValueError:
            raise ValueError(f"t_j = {t_j} is not a valid discrete time. See list of valid discrete times in AC_electrolytic_cell.t_array.")
        
        # Compute value of c
        cs = np.zeros(self.num_species * 2)
        for k in range(self.num_species * 2):
            contribution = self.coeffs[k] * self.fund_solns[j,k] * np.exp(self.char_expons[k] * t_j)    # Note that the TRUE t_j must be used in the exponent!
            if np.any(np.isnan(contribution)):
                continue
            else:
                cs = cs + contribution
        return cs    
    
    def c_s(self, t_j, species, bulk=True):
        """
        Finds the concentration of a particular species for discrete time t_j.
        
        Attributes:
            t_j: Discrete time
            species: String containing species name
            bulk: True for bulk concentration, False for surface concentration. Defaults to True.
        """
        
        cs = self.cs(t_j)
        try:
            species_s = self.species.index(species)
        except ValueError:
            raise ValueError(f"{species} not defined in species.")
        if bulk:
            return cs[2*species_s+1]
        else:
            return cs[2*species_s]
    
    def c_ss_s(self, species, bulk=True):
        """
        Finds the steady-state concentration of a particular species.
        
        Attributes:
            species: String containing species name
            bulk: True for bulk concentration, False for surface concentration. Defaults to True.
        """
        
        try:
            species_s = self.species.index(species)
        except ValueError:
            raise ValueError(f"{species} not defined in species.")
        if bulk:
            return self.c_ss[2*species_s+1]
        else:
            return self.c_ss[2*species_s]
        
    def c_ss_ts(self):
        """
        Finds the time evolution of the steady-state c-vector over a single oscillation period.
        
        Attributes:
            None
        """
        
        c_ss_ts = self.fund_solns[:,-1,:] * self.coeffs[-1]
        c_ss_ts = c_ss_ts.real
        return c_ss_ts

    def c_ss_t_s(self, species, bulk=True):
        """
        Finds the time evolution of the steady-state concentration of a particular species over a single oscillation period.
        
        Attributes:
            species: String containing species name
            bulk: True for bulk concentration, False for surface concentration. Defaults to True.
        """
        
        try:
            species_s = self.species.index(species)
        except ValueError:
            raise ValueError(f"{species} not defined in species.")
        if bulk:
            c_ss_ts = self.c_ss_ts()
            return c_ss_ts[:,2*species_s+1]
        else:
            c_ss_ts = self.c_ss_ts()
            return c_ss_ts[:,2*species_s]
    
    def current(self, t_j):
        """
        Finds the Faradaic current for discrete time t_j.
        
        Attributes:
            t_j: Discrete time
        """
        
        # Find the potential at time t
        E = self.E(t_j)
        
        # Find the c-vector at t_j
        cs = self.cs(t_j)
        
        current = 0
        for oxidant, reductant, k_O_R, alpha_O_R, n_O_R, E_O_R in self.electro_rxns:
            # Index the chemical species
            oxidant_s, reductant_s = self.species.index(oxidant), self.species.index(reductant)
            
            # Find the surface concentrations
            c_s_oxidant, c_s_reductant = cs[2*oxidant_s], cs[2*reductant_s]
            
            # Compute the contribution to the current
            current += n_O_R * k_O_R * \
                (np.exp(-alpha_O_R * n_O_R * self.f * (E - E_O_R)) * c_s_oxidant - \
                 np.exp((1 - alpha_O_R) * n_O_R * self.f * (E - E_O_R)) * c_s_reductant)
        
        current *= self.F * self.A
        return current
    
    def ave_current(self, t_start):
        """
        Finds the Faradaic current time-averaged over one oscillation period.
        
        Attributes:
            t_start: Starting time of the desired oscillation period. Must be integer multiples of the oscillation period.
        """
        if not AC_electrolytic_cell.test_if_equal(t_start * self.freq, np.round(t_start * self.freq), self.num_tol):
            raise IndexError(f"t_start = {t_start} is not an integer multiple of the oscillation period.")
        else:
            return np.average([self.current(t_start + t_j) for t_j in self.t_array])
        
    
    #####################################
    ### Functions that generate plots ###
    #####################################
    
    def plot_summary(self, subdir, filename, figsize=(8,8), cmap="viridis"):
        """
        Prepares a summary plot comprising
            (1) the concentrations of each species and 
            (2) the Faradaic currents time-averaged over one oscillation period
        at the characteristic decay timescales.
        These timescales are rounded off to the nearest oscillation period to avoid contributions from imaginary components.
        
        
        Attributes:
            subdir: Subdirectory to which the results should be written
            filename: Filename to which the results should be written
            figsize: Tuple describing the figure dimensions. Defaults to (8,8).
            cmap: String describing the colour map. Defaults to "viridis".
        """
        
        # Prepare filenames
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        # Organise data into a pandas dataframe
        adjusted_decay_ts, concs, currents = list(), list(), list()
        for char_expon in self.char_expons[:-1]:    # Skip the last characteristic exponent because it should be zero
            # Find characteristic decay timescale
            decay_t = -1 / char_expon.real  # Keep only the real part
            adjusted_decay_t = np.round(decay_t * self.freq) / self.freq     # Round timescale off to nearest oscillation period
            adjusted_decay_ts.append(f"{adjusted_decay_t:10.4f}")
            
            # Find concentrations
            cs = self.cs(adjusted_decay_t).real  # Keep only the real part
            for c_s in cs:
                concs.append(c_s)
                
            # Find current
            currents.append(self.ave_current(adjusted_decay_t).real * 1E3)  # Keep only the real part and convert to units of mA
            
        barplot_data = pd.DataFrame({"Time / s": [t for t in adjusted_decay_ts for count in range(2*self.num_species)],     # Every timescale has 2*num_species entries
                              "Species": [species + position for species in self.species for position in (" surf", " bulk")] * len(adjusted_decay_ts),
                              "Concentration / mM": concs})     # mM is also the SI unit for concentrations
        
        lineplot_data = pd.DataFrame({"Time / s": adjusted_decay_ts,
                              "Current / mA": currents})

        # Prepare colours
        cmap = cm.get_cmap(cmap)
        colors = [cmap(j / (self.num_species)) for j in range(self.num_species) for count in range(2)]
        
        # Plot data
        warnings.filterwarnings("ignore", category=FutureWarning)   # Unfortunately, not all Matplotlib versions support the update from "ci=None" to "errorbar=None", hence that error will be suppressed.
        fig, ax1 = plt.subplots(1, figsize = figsize, dpi = 300)
        sns.lineplot(data=lineplot_data, x="Time / s", y="Current / mA", marker="o", color="black", sort=False, ci=None, ax=ax1)
        ax2 = ax1.twinx()
        sns.barplot(data=barplot_data, x="Time / s", y="Concentration / mM", hue="Species", alpha=0.5, palette=colors, ci=None, ax=ax2)
        
        # Shade bars
        num_of_timescales = len(ax2.containers[0])
        hatches = itertools.cycle(['x', '*'])
        # hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
        for count, bar in enumerate(ax2.patches):
                # Seaborn plots each hue before moving on to the next hue
                # So we use the same shade until all timescales for a particular species have been shaded
                if count % num_of_timescales == 0:
                    hatch = next(hatches)
                bar.set_hatch(hatch)
        
        # Some aesthetic stuff
        plt.title('Summary plot', y=1.1, fontsize = 16)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=self.num_species, fancybox=True, shadow=True)
        
        # for ax_num,ax in enumerate(axes.flat):
        # Thicken plot borders
        [i.set_linewidth(2) for i in ax1.spines.values()]
        ax1.tick_params(width=2)
        
        # # Plot legend
        # ax1.legend()
        
        # # Set range of y-axis
        # ax.set_ylim([0.865,0.87])
        
        # # Set range of x-axis
        # ax.set_xlim([0.0,0.2])
        
        # Save plot
        fig.tight_layout()
        fig.savefig(filename_full +'.summary.png', bbox_inches='tight')
        fig.show()
        plt.close()
        
    def plot_fund_solns(self, subdir, filename, figsize=(8,8), cmap="viridis"):
        """
        Plots the fundamental solutions at t=0 multiplied by the expansion coefficients.
        Only the real components are presented.
        The decay timescales are rounded off to the nearest oscillation period.
        
        
        Attributes:
            subdir: Subdirectory to which the results should be written
            filename: Filename to which the results should be written
            figsize: Tuple describing the figure dimensions. Defaults to (8,8).
            cmap: String describing the colour map. Defaults to "viridis".
        """
        
        # Prepare filenames
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        # Organise data into a pandas dataframe
        adjusted_decay_ts, concs = list(), list()
        for k, char_expon in enumerate(self.char_expons[:-1]):    # Skip the last characteristic exponent because it should be zero
            # Find characteristic decay timescale
            decay_t = -1 / char_expon.real  # Keep only the real part
            adjusted_decay_t = np.round(decay_t * self.freq) / self.freq    # Round timescale off to nearest oscillation period
            adjusted_decay_ts.append(f"{adjusted_decay_t:10.4f}")
            
            # Find concentrations
            cs = (self.fund_solns[0,k] * self.coeffs[k]).real  # Keep only the real part
            for c_s in cs:
                concs.append(c_s)
                            
        barplot_data = pd.DataFrame({"Time / s": [t for t in adjusted_decay_ts for count in range(2*self.num_species)],     # Every timescale has 2*num_species entries
                              "Species": [species + position for species in self.species for position in (" surf", " bulk")] * len(adjusted_decay_ts),
                              "Concentration / mM": concs})     # mM is also the SI unit for concentrations
        
        # Prepare colours
        cmap = cm.get_cmap(cmap)
        colors = [cmap(j / (self.num_species)) for j in range(self.num_species) for count in range(2)]
        
        # Plot data
        warnings.filterwarnings("ignore", category=FutureWarning)   # Unfortunately, not all Matplotlib versions support the update from "ci=None" to "errorbar=None", hence that error will be suppressed.
        fig, ax1 = plt.subplots(1, figsize = figsize, dpi = 300)
        sns.barplot(data=barplot_data, x="Time / s", y="Concentration / mM", hue="Species", alpha=0.5, palette=colors, ci=None, ax=ax1)
        
        # Shade bars
        num_of_timescales = len(ax1.containers[0])
        hatches = itertools.cycle(['x', '*'])
        # hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
        for count, bar in enumerate(ax1.patches):
                # Seaborn plots each hue before moving on to the next hue
                # So we use the same shade until all timescales for a particular species have been shaded
                if count % num_of_timescales == 0:
                    hatch = next(hatches)
                bar.set_hatch(hatch)
        
        # Some aesthetic stuff
        plt.title(r'Fundamental solutions at $t=0$', y=1.1, fontsize = 16)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=self.num_species, fancybox=True, shadow=True)
        
        # for ax_num,ax in enumerate(axes.flat):
        # Thicken plot borders
        [i.set_linewidth(2) for i in ax1.spines.values()]
        ax1.tick_params(width=2)
        
        # # Plot legend
        # ax1.legend()
        
        # # Set range of y-axis
        # ax.set_ylim([0.865,0.87])
        
        # # Set range of x-axis
        # ax.set_xlim([0.0,0.2])
        
        # Save plot
        fig.tight_layout()
        fig.savefig(filename_full +'.fund_solns.png', bbox_inches='tight')
        fig.show()
        plt.close()