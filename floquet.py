# -*- coding: utf-8 -*-

import os
import sys
import platform
import warnings
import numpy as np
import scipy as sp
import dill
import pathos.multiprocessing as mp
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

class Floquet_solver:
    """
    Solves a differential equation of Floquet form.
    
    Assumes a differential equation of the form dx/dt = A(t) x(t) with x(t) being a vector of length N and
    A(t) being a NxN matrix that is periodic in time with frequency freq. Also assumes an initial value problem.
    
    Attributes:
        A: Function generating the NxN rate constant matrix that is periodic in time. 
            Function should call time as its argument and return a numpy array.
        freq: Frequency of periodicity
        x_0: Initial x. Should be a numpy vector of length N.
        
        **kwargs
        min_num_int: Minimum number of intervals to divide the period into during numerical integration.
                        If this value is too small, solver will keep increasing it in multiples of 10 until a suitable value is found.
                        However, only min_num_int timepoints will be stored.
                        Defaults to 100.
        guess_num_int: Guess number of intervals to divide the period into during numerical integration.
                        Must be an integer multiple of min_num_int.
                        Defaults to min_num_int.
        tol: Tolerance for any dimensionless error. Defaults to 1E-2.
        num_procs: Number of parallel processes to perform. Defaults to 1.
        method: Method for finding the one-period time propagator. 
                    1 = Expanding the exponential (old algorithm).
                    2 = Not expanding the exponential (new algorithm).
                    Defaults to 2.
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
            if not Floquet_solver.test_if_equal(a, b, tol):
                return False
        return True

    @staticmethod
    def test_if_matrices_equal(A, B, tol):
        """
        Tests if matrices A and B are nearly equal using test_if_equal function.
        
        Attributes:
            A, B: Numpy matrices to be compared
            tol: Tolerance for error
            
        Returns:
            Boolean describing test result.
        """
        
        for row in np.dstack((A,B)):
            for (a,b) in row:
                if not Floquet_solver.test_if_equal(a, b, tol):
                    return False
        return True    
    
    def __init__(self,
                 A,
                 freq: float,
                 x_0: np.ndarray,
                 **kwargs):
        self.A = A
        self.freq = freq
        self.x_0 = x_0
        try:
            self.min_num_int = kwargs["min_num_int"]
        except KeyError:
            self.min_num_int = 100
        try:
            self.guess_num_int = kwargs["guess_num_int"]
        except KeyError:
            self.guess_num_int = self.min_num_int
        try:
            self.tol = kwargs["tol"]
        except KeyError:
            self.tol = 1E-2
        try:
            self.num_procs = kwargs["num_procs"]
        except KeyError:
            self.num_procs = 1
        try:
            self.method = kwargs["method"]
        except KeyError:
            self.method = 2
        
        # Check that all inputs are valid
        if not callable(self.A):
            raise TypeError(f"A should be a function. Type of A: {type(self.A)}.")
        if type(self.A(0)) != np.ndarray:
            raise TypeError(f"A should return a numpy array. Type of A(0): {type(self.A(0))}.")
        if self.A(0).ndim != 2:
            raise TypeError(f"A should return an NxN matrix. Dimensions of A(0): {self.A(0).ndim}.")
        if self.x_0.ndim != 1:
            raise TypeError(f"x_0 should be a vector of length N. Dimensions of x_0: {self.x_0.ndim}.")
        if self.A(0).shape != (self.x_0.shape[0],self.x_0.shape[0]):
            raise TypeError(f"A should return an NxN matrix and x_0 should be a vector of length N. Shape of A(0): {self.A(0).shape}. Shape of x_0: {self.x_0.shape}.")
        if not Floquet_solver.test_if_matrices_equal(self.A(1 / self.freq), self.A(0), self.tol):
            raise TypeError(f"A should be periodic with frequency freq.")
        if type(self.min_num_int) != int:
            raise TypeError(f"min_num_int should be an integer. Type of min_num_int: {type(self.min_num_int)}.")
        if type(self.guess_num_int) != int:
            raise TypeError(f"guess_num_int should be an integer. Type of guess_num_int: {type(self.guess_num_int)}.")
        if self.guess_num_int % self.min_num_int:
            raise TypeError(f"guess_num_int must be divisible by min_num_int. guess_num_int: {self.guess_num_int}. min_num_int: {self.min_num_int}")
        if type(self.tol) != float:
            raise TypeError(f"tol should be a float. Type of tol: {type(self.tol)}.")
        if type(self.num_procs) != int:
            raise TypeError(f"num_procs should be an integer. Type of num_procs: {type(self.num_procs)}.")
        if type(self.method) != int:
            raise TypeError(f"method should be an integer. Type of method: {type(self.method)}.")
            
        # Compute some universal parameters
        self.N = self.x_0.shape[0]

    def solve(self):
        """
        Solves the differential equation by diagonalising the one-period time propagator.
        
        Solutions are stored as such: 
            fund_solns: Fundamental solutions. First index labels the time, next two indices label the solution matrices.
            char_expons: Characteristic exponents
            coeffs: Expansion coefficients
            t_array: Discretised time array
        
        Attributes:
            None
        """
        
        # Define desired time array
        self.t_array = np.linspace(0, 1 / self.freq, self.min_num_int + 1)  
        
        if self.method == 1:
            # Prepare approximate one-period time propagator
            print("Preparing approximate one-period time propagator...")
            print(f"Dividing into {self.num_procs} parallel processes...")
            num_int = self.guess_num_int
            print(f"Starting with {num_int} intervals...")
            Adt_is_large = True
            while Adt_is_large:
                # Because we are going to use "num_int // self.min_num_int" everywhere in this code, we will first check that this division has no remainder.
                if num_int % self.min_num_int: raise TypeError(f"num_int must be divisible by min_num_int. num_int: {num_int}. min_num_int: {self.min_num_int}")
                
                # Prepare time step
                dt = 1 / self.freq / num_int
                
                # Prepare an empty array to store desired intermediate time propagators
                time_propagators = np.zeros((self.min_num_int + 1, self.N, self.N), dtype=np.float128)
                time_propagators[0] = np.identity(self.N, dtype=np.float128)   # Initialise array
                
                # Divide 0 <= j < num_int into self.min_num_int parts and perform matrix multiplication within each part
                proc_indices = np.arange(0, self.min_num_int, 1, dtype=int)
                j_domains = [(proc_index * (num_int // self.min_num_int), (proc_index + 1) * (num_int // self.min_num_int)) for proc_index in proc_indices]
                
                # Define parallel operation
                def operation(proc_index):
                    start_j, stop_j = j_domains[proc_index]
                    
                    # Compute one-period time propagator within j domain
                    time_propagator_j = np.identity(self.N, dtype=np.float128)
                    j = start_j
                    
                    while j < stop_j:
                        t_j = j * dt
                        Adt_j = self.A(t_j) * dt
                        Adt_j = Adt_j.astype(np.float128)
                        
                        # If the finite difference is too large, then further divide the oscillation period.
                        if not np.all(np.abs(Adt_j) < self.tol):
                            raise Exception(f"The number of intervals is too small for finite difference to work.")   
                        else:
                            time_propagator_j = (Adt_j + np.identity(self.N, dtype=np.float128)).dot(time_propagator_j)    # Need to "dot" from final time to initial time
                            j += 1
                        
                    sys.stdout.flush()      # Flush all output from the process
                    return time_propagator_j
                
                # Run parallel operations
                try:
                    procs_pool = mp.Pool(self.num_procs)
                    matrix_mults = procs_pool.map(operation, proc_indices)
                    procs_pool.close()  # Close the pool
                    procs_pool.terminate()  # Kill the pool
                except Exception:
                    procs_pool.terminate()  # Kill the pool
    
                    # If the finite difference is too large, then further divide the oscillation period.
                    # To prevent an infinite loop from occurring, only 10 different num_int values are attempted.
                    if np.log10(num_int // self.min_num_int) >= 9:
                        raise Exception("The finite difference method did not converge within 10 tries. Try with a higher guess_num_int.")
                    else:
                        num_int *= 10
                        print(f"The number of intervals is too small for finite difference to work. Trying with {num_int} intervals...")
                        continue
    
                # If the finite difference method is successful, compute time propagators and exit the while loop.
                for count, matrix_mult in enumerate(matrix_mults):
                    time_propagators[count+1] = matrix_mult.dot(time_propagators[count])
                
                print(f"Success! The oscillation period was divided into {num_int} intervals.")
                period_propagator = time_propagators[-1]
                Adt_is_large = False
            
            # Diagonalise approximate one-period time propagator
            print("Diagonalising the approximate one-period time propagator...")
            period_propagator = period_propagator.astype(np.float64)    # np.linalg.eig does not support np.float128
            eigenvalues, eigenvectors_T = np.linalg.eig(period_propagator)
            fund_solns_0 = eigenvectors_T.T
            if np.any(eigenvalues == 0):
                print("* Reminder: At least one of the characteristic exponents is zero, hence some RuntimeWarnings are expected. They are harmless and will be suppressed. *")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            char_expons = np.emath.log(eigenvalues) * self.freq  # Returns the principal value of complex logarithm
            
            # Clean up infinities and NaNs in char_expons
            if char_expons.dtype == complex:
                char_expons = np.nan_to_num(char_expons.real) + np.nan_to_num(char_expons.imag) * 1j
            else:
                char_expons = np.nan_to_num(char_expons)
            
            # Sort by real parts of the characteristic exponents
            sort_indices = np.argsort(char_expons.real)
            char_expons, fund_solns_0 = char_expons[sort_indices], fund_solns_0[sort_indices]
        
        elif self.method == 2:
            print("*** Using the new algorithm to obtain the one-period time propagator ***")
            
            # Prepare approximate one-period time propagator
            print("Preparing approximate one-period time propagator...")
            print(f"Dividing into {self.num_procs} parallel processes...")
            num_int = self.guess_num_int
            print(f"Starting with {num_int} intervals...")
            converged = False
            while not converged:
                # Because we are going to use "num_int // self.min_num_int" everywhere in this code, we will first check that this division has no remainder.
                if num_int % self.min_num_int: raise TypeError(f"num_int must be divisible by min_num_int. num_int: {num_int}. min_num_int: {self.min_num_int}")
                
                # Prepare time step
                dt = 1 / self.freq / num_int
                
                # Prepare an empty array to store desired intermediate time propagators
                time_propagators = np.zeros((self.min_num_int + 1, self.N, self.N), dtype=np.float128)
                time_propagators[0] = np.identity(self.N, dtype=np.float128)   # Initialise array
                
                # Divide 0 <= j < num_int into self.min_num_int parts and perform matrix multiplication within each part
                proc_indices = np.arange(0, self.min_num_int, 1, dtype=int)
                j_domains = [(proc_index * (num_int // self.min_num_int), (proc_index + 1) * (num_int // self.min_num_int)) for proc_index in proc_indices]
                
                # Define parallel operation
                def operation(proc_index):
                    start_j, stop_j = j_domains[proc_index]
                    
                    # Compute one-period time propagator within j domain
                    time_propagator_j = np.identity(self.N, dtype=np.float128)
                    j = start_j
                    
                    while j < stop_j:
                        t_j = j * dt
                        Adt_j = self.A(t_j) * dt
                        Adt_j = Adt_j.astype(np.float64)    # Unfortunately, the cluster's sp.linalg.expm does not support float128
                        
                        time_propagator_j = sp.linalg.expm(Adt_j).dot(time_propagator_j)    # Need to "dot" from final time to initial time
                        j += 1
                        
                    sys.stdout.flush()      # Flush all output from the process
                    return time_propagator_j
                
                # Run parallel operations
                procs_pool = mp.Pool(self.num_procs)
                matrix_mults = procs_pool.map(operation, proc_indices)
                procs_pool.close()  # Close the pool
                procs_pool.terminate()  # Kill the pool
    
                # Compute time propagators
                for count, matrix_mult in enumerate(matrix_mults):
                    time_propagators[count+1] = matrix_mult.dot(time_propagators[count])
                    
                period_propagator = time_propagators[-1]
                
                # Diagonalise approximate one-period time propagator
                print("Diagonalising the approximate one-period time propagator...")
                period_propagator = period_propagator.astype(np.float64)    # np.linalg.eig does not support np.float128
                eigenvalues, eigenvectors_T = np.linalg.eig(period_propagator)
                fund_solns_0 = eigenvectors_T.T
                if np.any(eigenvalues == 0):
                    print("* Reminder: At least one of the characteristic exponents is zero, hence some RuntimeWarnings are expected. They are harmless and will be suppressed. *")
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                char_expons = np.emath.log(eigenvalues) * self.freq  # Returns the principal value of complex logarithm
                
                # Clean up infinities and NaNs in char_expons
                if char_expons.dtype == complex:
                    char_expons = np.nan_to_num(char_expons.real) + np.nan_to_num(char_expons.imag) * 1j
                else:
                    char_expons = np.nan_to_num(char_expons)
                
                # Sort by real parts of the characteristic exponents
                sort_indices = np.argsort(char_expons.real)
                char_expons, fund_solns_0 = char_expons[sort_indices], fund_solns_0[sort_indices]
                
                # If real parts of characteristic exponents are close to the previous iteration, exit the while loop.
                # Otherwise, further divide the oscillation period.
                try:
                    converged = Floquet_solver.test_if_vectors_equal(char_expons.real, char_expons_old.real, self.tol)
                        
                except NameError:
                    num_int *= 10
                    char_expons_old = char_expons
                    print(f"Since this is the first result, we will do one more computation with {num_int} intervals to check for convergence...")
                    continue
                
                if converged:
                    print(f"Success! The oscillation period was divided into {num_int} intervals.")
                
                else:
                    # To prevent an infinite loop from occurring, only 10 different num_int values are attempted.
                    if np.log10(num_int // self.min_num_int) >= 9:
                        raise Exception("The finite difference method did not converge within 10 tries. Try with a higher guess_num_int.")
                    else:
                        num_int *= 10
                        char_expons_old = char_expons
                        print(f"The number of intervals is too small for finite difference to work. Trying with {num_int} intervals...")
                        continue
            
        else:
            raise Exception(f"Method {self.method} has not been implemented yet.")
        
        # Find fundamental solutions for all times t
        print("Finding fundamental solutions for all times t...")
        # Prepare an empty array to store desired intermediate time solutions
        # Note that dtype must be set to complex if complex fundamental solutions are present
        fund_solns = np.zeros((self.min_num_int + 1, self.N, self.N), dtype=char_expons.dtype)
        for count, t in enumerate(self.t_array):
            time_propagator_count = time_propagators[count]
            for k, p_k_0 in enumerate(fund_solns_0):
                char_expon = char_expons[k]
                j = count * (num_int // self.min_num_int)     # Map count (index in desired time array) to j (index in finite difference method)
                fund_solns[count,k] = time_propagator_count.dot(p_k_0) * np.exp(-char_expon * j * dt) 

        # Find expansion coefficients
        print("Finding expansion coefficients...")
        coeffs = np.linalg.solve(fund_solns_0.T, self.x_0)
        # Check if solution is correct
        if not np.allclose(np.dot(fund_solns_0.T, coeffs), self.x_0):
            mat_det = np.linalg.det(fund_solns_0.T)
            raise Exception(f"np.linalg.inv did not work. Matrix might be singular. Matrix determinant = {mat_det}")

        # Storing all data
        print("Storing data...")
        self.fund_solns, self.char_expons, self.coeffs = fund_solns, char_expons, coeffs
        
        print("\n")
        print("Done! Solutions are stored as such: ")
        print("\t fund_solns: Fundamental solutions. First index labels the time, next two indices label the solution matrices.")
        print("\t char_expons: Characteristic exponents")
        print("\t coeffs: Expansion coefficients")
        print("\t t_array: Discretised time array")
        print("\n")
        
    def x(self, t_j):
        """
        Finds solution for discrete time t_j.
        
        Attributes:
            t_j: Discrete time
        """
        
        # Find the equivalent t_j in the first time period
        adjusted_t_j = np.abs(t_j % (1 / self.freq))
        
        # Check if t_j is a valid discrete time within some error
        try: 
            truth_matrix = (self.t_array >= adjusted_t_j - self.tol * adjusted_t_j) & (self.t_array <= adjusted_t_j + self.tol * adjusted_t_j)
            j = np.where(truth_matrix == True)[0][0]
        except IndexError:
            raise IndexError(f"t_j = {t_j} is not a valid discrete time. See list of valid discrete times in Floquet_solver.t_array.")
        
        # Compute value of x
        x = np.zeros(self.N)
        for k in range(self.N):
            x = x + self.coeffs[k] * self.fund_solns[j,k] * np.exp(self.char_expons[k] * t_j)     # Note that the TRUE t_j must be used in the exponent!
        
        return x
    
    @staticmethod
    def benchmark():
        """
        Benchmarks solver against a known result.
        
        Attributes:
            None
        """
        
        # Prepare known solutions
        def x_true(t, c_1, c_2):
            return np.array([c_1 * np.exp(1) * np.exp(-np.cos(t)) + c_2 * np.exp(-np.cos(t)) * sp.integrate.quad(lambda x: np.exp(np.cos(x)), 0, t)[0],
                             c_1 * np.exp(1) * np.sin(t) * np.exp(-np.cos(t)) + c_2 * (np.sin(t) * np.exp(-np.cos(t)) * sp.integrate.quad(lambda x: np.exp(np.cos(x)), 0, t)[0] + 1)])
        
        # Construct inputs for c_1 = 1, c_2 = 2
        def A(t):
            return np.array([[0, 1],
                             [np.cos(t), np.sin(t)]])
        freq = 1 / (2 * np.pi)
        x_0 = x_true(0, c_1=1, c_2=2)
        
        # Solve for c_1 = 1, c_2 = 2
        solver = Floquet_solver(A, freq, x_0)
        solver.solve()
        
        # Check solutions
        test_t_j = np.arange(0, 2 / freq , 1 / freq / 4)
        print("-------------------------------------------BENCHMARKING RESULTS----------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print(" \t\t Time \t\t\t Numerical solutions \t\t\t Exact solutions \t\t\t Error")
        print("-------------------------------------------------------------------------------------------------------------")
        for t_j in test_t_j:
            numerical, exact = solver.x(t_j), x_true(t_j, c_1=1, c_2=2)
            error = numerical / exact - 1
            print(f" \t {t_j:10.4f} \t\t {numerical[0]:10.4f} \t {numerical[1]:10.4f} \t\t {exact[0]:10.4f} \t {exact[1]:10.4f} \t\t {error[0]:10.4f} \t {error[0]:10.4f}")
        print("-------------------------------------------------------------------------------------------------------------")
         
if __name__ == '__main__':
    Floquet_solver.benchmark()