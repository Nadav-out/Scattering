from unittest import result
import numpy as np
from scipy.interpolate import interp1d
import sympy as sym
from  scipy.optimize import root


class Classical:
    def __init__(self, v_tilde, x_sym, beta_arr=None, order=1, xT_helper_arr=None,
                 beta_max_ratio_log=6, n_beta=1000, n_x=1000,log_z_range=None, n_z=1000,
                 n_beta_small=100, n_xT_small=150, n_beta_large=70, n_xT_large=200):

        if beta_arr is None:
            beta_arr = np.logspace(-8,8,100)
        if xT_helper_arr is None:
            xT_helper_arr = np.linspace(0, 100, 100)
        if log_z_range is None:
            log_z_range = [-10,4]

        # Parameters used in 'get_symbolic_expressions'
        self.v_tilde = v_tilde
        self.x_sym = x_sym
        self.order = order

        # Parameter used in 'find_beta_c_and_xT_c_int'
        self.xT_helper_arr=xT_helper_arr

        # Parameters used in 'find_xT_1_int'
        self.n_x = n_x
        self.n_beta = n_beta
        self.beta_max_ratio_log=beta_max_ratio_log

        # Parameters used in 'get_phi_arr'
        self.log_z_range=log_z_range
        self.n_z=n_z

        # Parameters used in 'compute'
        self.n_beta_small=n_beta_small
        self.n_xT_small=n_xT_small      
        self.n_beta_large=n_beta_large
        self.n_xT_large=n_xT_large

        # Range of beta for computation.
        self.beta_arr=beta_arr

        
        def Texp(arr):
            arr = np.asarray(arr, dtype=float) # Ensure arr is a numpy array with float type
            result = np.empty_like(arr, dtype=float)

            # Conditions
            result[arr < -500] = 0
            result[arr > 500] = np.inf
            result[(arr > -0.001) & (arr < 0.001)] = 1 + arr[(arr > -0.001) & (arr < 0.001)]
            mask = (arr >= 0.001) & (arr <= 500) | (arr > -500) & (arr <= -0.001)
            result[mask] = np.exp(arr[mask])
            

            return result
        # create a new namespace with numpy and your Texp function
        self.namespace = {**np.__dict__, 'exp': Texp}


        # Checks if the potential is attractive
        self.is_attractive=np.max(sym.lambdify(self.x_sym, self.v_tilde,"numpy")(np.logspace(-2,2,20)))<=0

        if self.is_attractive:
            print('The potential is attractive, building classical deflection angles tables.')
            self.d_phi_dx_lam, self.y_sq_sol_lam, self.jacobian_y_sq_xT_lam, self.beta_c_sol = self.get_symbolic_expressions()
            self.beta_c, self.xT_c_of_beta_int = self.find_beta_c_and_xT_c_int()
            self.xT_1_of_beta_int = self.find_xT_1_int()

        else:
            print('The potential is repulsive, building classical deflection angles tables.')
            self.beta_arr, self.xT_min_arr=self.xT_min_of_beta_arr()
            self.d_phi_dx_lam, self.jacobian_y_sq_xT_lam = self.get_symbolic_expressions_repulsive()
            

            
        

    

    def get_symbolic_expressions_repulsive(self):
        

         # Define additional symbols
        xT_sym, beta_sym = sym.symbols('xT_sym beta_sym', positive=True)

        # Build the integrand terms
        V=self.v_tilde/self.x_sym
        # Define y_sq_sol(xT, beta), the solution for U(xT, y, beta) = 1
        y_sq=xT_sym**2*(1-beta_sym*V.subs(self.x_sym, xT_sym))    
        T1 = y_sq / self.x_sym**2
        U = T1 + beta_sym * V
        pole_sq = (self.x_sym - xT_sym)**.5


        # Manually integrate by parts n=order times
        n = self.order
        g_n = ((-2)**n / sym.factorial2(2*n - 1)) * pole_sq**(2*n - 1)
        f0_denom = self.x_sym**2 * (1 - U)**.5
        f0 = pole_sq / f0_denom
        f_n = f0.diff(self.x_sym, n)

        # The modified integrand
        d_phi_dx = g_n * f_n * y_sq**.5

        # To not ever preforme y integration, we should replace 2*pi*y*dy-->pi*|dy^2/dxT|dxT
        jacobian_y_sq_xT=y_sq.diff(xT_sym)#(y_sq_sol.diff(xT_sym)**2)**.5

        # Convert symbolic expressions to numpy functions
        d_phi_dx_lam = sym.lambdify((self.x_sym, xT_sym, beta_sym), d_phi_dx, self.namespace)
        jacobian_y_sq_xT_lam=sym.lambdify((xT_sym, beta_sym), jacobian_y_sq_xT, self.namespace)
        
        

        return d_phi_dx_lam, jacobian_y_sq_xT_lam
    
    def xT_min_of_beta_arr(self):
        """
        Finds the minimum xT value for each beta value in beta_arr in the repulsive case.

        Returns:
        ------------
        xT_min_arr (numpy.arry)   : An array of minimum xT values for each beta value in beta_arr.
        """
        
        # beta(xT) lambda function, as calculated int `get_symbolic_expressions_small_repulsive'
        beta_xT_lam=sym.lambdify(self.x_sym,sym.simplify(self.x_sym/self.v_tilde),self.namespace)
        
        # initial range of xT, which is used to find beta(xT)
        beta_min=self.beta_arr[0]
        beta_max=self.beta_arr[-1]
        xT_max=np.log10(1+beta_max)
        xT_min=np.log10(1+beta_min)
        # make sure we cover the whole range 
        while beta_xT_lam(xT_max)<self.beta_arr[-1]:
            xT_max=xT_max+np.log(1.1)
        while beta_xT_lam(xT_min)>self.beta_arr[0]:
            xT_min=xT_min/1.1

        

        # fill in the array XXX Come back to this!
        # fill in the array
        xT_arr_left = np.linspace(xT_min, 1, 5000)
        xT_arr_right = np.geomspace(1, xT_max, 5000)

        beta_xT_left = beta_xT_lam(xT_arr_left)
        beta_xT_right = beta_xT_lam(xT_arr_right)

        # create interpolation functions
        interp_func_left = interp1d(np.log10(beta_xT_left), np.log10(xT_arr_left), kind='linear')
        interp_func_right = interp1d(np.log10(beta_xT_right), xT_arr_right, kind='linear')

        # split beta_arr
        beta_arr_left = self.beta_arr[self.beta_arr <=beta_xT_left[-1]]
        beta_arr_right = self.beta_arr[self.beta_arr >= beta_xT_right[0]]

        # apply the appropriate interpolation function to each part of beta_arr
        xT_arr_left = 10**interp_func_left(np.log10(beta_arr_left))
        xT_arr_right = interp_func_right(np.log10(beta_arr_right))

        # concatenate the left and right parts
        xT_arr = np.concatenate((xT_arr_left, xT_arr_right))
        

        beta_xT = beta_xT_lam(xT_arr)

        # xT_min_arr=np.concatenate((beta_low,xT_arr_low))

        

        return beta_xT, xT_arr




    def get_symbolic_expressions(self):
        """
        Generates a tuple of numpy functions derived from a symbolic potential, including the derivative of the classical deflection angle phi, the impact parameter, and the critical impact parameter.
        
        Returns:
        ------------
        [numpy function] d_phi_dx_lam(x, xT, beta)      : The differential classical deflection angle phi.
        [numpy function] y_sq_sol_lam(xT, beta)         : The impact parameter^2.
        [numpy function] jacobian_y_sq_xT_lam(xT, beta) : The Jacobian |dy^2/dxT|.
        [sympy expression] beta_c_sol                   : The beta value solving U'(xT)=0.
        """

             # Define additional symbols
        xT_sym, beta_sym = sym.symbols('xT_sym beta_sym', positive=True)

        # Build the integrand terms
        V=self.v_tilde/self.x_sym
        # Define y_sq_sol(xT, beta), the solution for U(xT, y, beta) = 1
        y_sq=xT_sym**2*(1-beta_sym*V.subs(self.x_sym, xT_sym))    
        T1 = y_sq / self.x_sym**2
        U = T1 + beta_sym * V
        pole_sq = (self.x_sym - xT_sym)**.5


        # Manually integrate by parts n=order times
        n = self.order
        g_n = ((-2)**n / sym.factorial2(2*n - 1)) * pole_sq**(2*n - 1)
        f0_denom = self.x_sym**2 * (1 - U)**.5
        f0 = pole_sq / f0_denom
        f_n = f0.diff(self.x_sym, n)

        # The modified integrand
        d_phi_dx = g_n * f_n * y_sq**.5

        # To not ever preforme y integration, we should replace 2*pi*y*dy-->pi*|dy^2/dxT|dxT
        jacobian_y_sq_xT=y_sq.diff(xT_sym)#(y_sq_sol.diff(xT_sym)**2)**.5

        # Convert symbolic expressions to numpy functions
        d_phi_dx_lam = sym.lambdify((self.x_sym, xT_sym, beta_sym), d_phi_dx, self.namespace)
        jacobian_y_sq_xT_lam=sym.lambdify((xT_sym, beta_sym), jacobian_y_sq_xT, self.namespace)
        
        
        # Find beta_sol_sym, the beta value also solving dU/dx=0 at xT.
        beta_c_sol = sym.cancel(sym.solve(U.diff(self.x_sym), beta_sym)[0].subs(xT_sym,self.x_sym))

        # Convert symbolic expressions to numpy functions
        d_phi_dx_lam = sym.lambdify((self.x_sym, xT_sym, beta_sym), d_phi_dx_Y,"numpy")
        jacobian_y_sq_xT_lam=sym.lambdify((xT_sym, beta_sym), jacobian_y_sq_xT,"numpy")
        y_sq_sol_lam=sym.lambdify((xT_sym, beta_sym), y_sq_sol,"numpy")

        return d_phi_dx_lam, y_sq_sol_lam, jacobian_y_sq_xT_lam, beta_c_sol


    def find_beta_c_and_xT_c_int(self):
        """
        Finds the critical beta value and the corresponding xT_c.

        Returns:
        ------------
        beta_c (float)                                  : The critical beta value, at which U'(xT)=0 has a local maximum.
        xT_c_of_beta_int (scipy.interpolate.interp1d)   : An interpolating function for xT_c as a function of beta, using log-log interpolation.
        """

        # Compute derivative of self.beta_c_sol with respect to xT
        d_beta_c = sym.cancel(self.beta_c_sol.diff(self.x_sym))

        # Evaluate self.beta_c_sol and its derivative at the xT values
        beta_lis = sym.lambdify((self.x_sym), self.beta_c_sol, "numpy")(self.xT_helper_arr)
        d_beta_lis = sym.lambdify((self.x_sym), d_beta_c, "numpy")(self.xT_helper_arr)

        # Only consider positive beta values
        positive_beta_inds = beta_lis > 0
        beta_lis_new = beta_lis[positive_beta_inds]
        xT_helper_arr_new = self.xT_helper_arr[positive_beta_inds]
        d_beta_lis_new = d_beta_lis[positive_beta_inds]

        # Find where the derivative changes sign
        d_beta_sign_changes = np.where(np.diff(np.sign(d_beta_lis_new)))[0]

        # Check if any sign change is found
        if len(d_beta_sign_changes) > 1:
            warnings.warn("The results might not be correct for cases when the potential exhibits many features.\n(Multiple critical impact parameters found)")
        elif len(d_beta_sign_changes) < 1:
            print("No critical impact parameter found, assuming a repulsive potential")
            return None
        
        # The critical beta value is the first beta value where the derivative changes sign
        beta_c = beta_lis_new[d_beta_sign_changes[0]]
        print("The critical beta value is " ,beta_c)
        # Create an interpolating function for xT_c as a function of beta using log-log interpolation
        log_interp = interp1d(np.log10(beta_lis_new[d_beta_sign_changes[0]:]), np.log10(xT_helper_arr_new[d_beta_sign_changes[0]:]), bounds_error=False, fill_value=-np.inf)

        def xT_c_of_beta_int(beta):
            return 10**log_interp(np.log10(beta))
            
        return beta_c, xT_c_of_beta_int


    def find_xT_1_int(self):
        """
        Find xT_1 as a function of beta and return a function that interpolates these values.

        Returns:
        ------------
        xT_1_of_beta_int (scipy.interpolate.interp1d) : A function that interpolates xT_1(beta), where xT_1 is the point in the range [0, xT_c] that has the same Y value as xT_c.
        """

        # Generate beta values
        beta_max = self.beta_c * 10**self.beta_max_ratio_log
        beta_range = np.geomspace(self.beta_c, beta_max, self.n_beta)

        # Compute corresponding xT_c values
        xT_c_range = self.xT_c_of_beta_int(beta_range)

        # Generate xT values in the range [0, xT_c] for each beta
        x_arr = xT_c_range[:, np.newaxis] * np.logspace(-self.beta_max_ratio_log*2, -0.05, self.n_x)

        # Calculate Y^2 for each xT and beta
        y_sqr_arr = self.y_sq_sol_lam(x_arr, beta_range[:, np.newaxis])
        y_sqr_at_xTc = self.y_sq_sol_lam(xT_c_range, beta_range)

        # Find the xT_1 value for each beta that minimizes the squared difference of Y^2 - Y^2(xT_c)
        argsmin = np.argmin((y_sqr_arr - y_sqr_at_xTc[:, np.newaxis])**2, axis=1)
        xT_1_arr = x_arr[range(self.n_beta), argsmin]
        xT_1_arr[0] = xT_c_range[0]

        # Return an interpolating function for xT_1 as a function of beta
        def xT_1_of_beta_int(beta):
            return 10**interp1d(np.log10(beta_range), np.log10(xT_1_arr), bounds_error=False, fill_value=-np.inf)(np.log10(beta))
        
        return xT_1_of_beta_int


    def get_phi_arr(self,dphi_dx_func, beta_arr, xT_arr, is_meshgrid=False):
        """
        Generate the classical deflection angle (Phi) in a scattering of the potential v_tilde.

        Parameters:
        ------------
        dphi_dx_func (function) : A function to compute the deflection angle.
        beta_arr (numpy.ndarray): Array of beta values. Beta is the dimensionless interaction strength.
        xT_arr (numpy.ndarray)  : Array of xT values. xT is the classical turning point.
        log_z_range (list)      : List of two floats indicating the range for z values. z is a convenient grid for the integration.
        n_z (int)               : Number of z values.

        Returns:
        ------------
        numpy.ndarray : A 2D array of Phi values. Phi is the classical deflection angle.
        """

        # Generate logarithmically spaced arrays
        z_values = np.logspace(*self.log_z_range, self.n_z)

        if(is_meshgrid):
            beta_mesh, xT_mesh = beta_arr, xT_arr
        else:
            # Create xT and beta 2D meshgrids
            beta_mesh, xT_mesh = np.meshgrid(beta_arr, xT_arr,indexing='ij')

        # Create the 3D x meshgrid, over which dphi will be integrated
        x_mesh = xT_mesh[:,:,None] * (1 + z_values)


        # Apply dphi_dx_func and replace NaN values with 0
        dphi_dx = np.nan_to_num(dphi_dx_func(x_mesh, xT_mesh[:,:,None], beta_mesh[:,:,None]))
        
        # Compute the integral using the trapezoidal rule
        phi_arr = np.trapz(dphi_dx, x_mesh, axis=2)
        
        return phi_arr

    def compute(self):
        """
        Executes all necessary calculations.
        """
        # Attractive case
        if self.is_attractive:
            # define range of beta values for small beta regime
            beta_list_small_beta = self.beta_arr[self.beta_arr<=self.beta_c]
            
            # define range of xT values for small beta regime
            xT_list_small_beta = np.linspace(0.0001, 6, 150)
            
            # calculate the deflection angle for small beta regime
            phi_arr_small_beta = self.get_phi_arr(self.d_phi_dx_lam, beta_list_small_beta, xT_list_small_beta)
            
            # generate meshgrid for small beta regime
            beta_mesh_small_beta, xT_mesh_small_beta = np.meshgrid(beta_list_small_beta, xT_list_small_beta, indexing='ij')
            
            # calculate the Jacobian for small beta regime
            jac_arr_small_beta = self.jacobian_y_sq_xT_lam(xT_mesh_small_beta, beta_mesh_small_beta)
            
            # define range of beta values for large beta regime
            beta_list_lrg = self.beta_arr[self.beta_arr>self.beta_c]
            
            # define range of xT values for large beta regime
            x1_list_lrg = self.xT_1_of_beta_int(beta_list_lrg)
            xc_list_lrg = self.xT_c_of_beta_int(beta_list_lrg)
            
            # define z ranges for low and high regimes in large beta
            z_lis_low = np.linspace(0.001, 1, 200)
            z_lis_high = np.logspace(0.001, 1, 150)
            
            # generate meshgrids for low and high regimes in large beta
            beta_mesh_low, z_mesh_low = np.meshgrid(beta_list_lrg, z_lis_low, indexing='ij')
            beta_mesh_high, z_mesh_high = np.meshgrid(beta_list_lrg, z_lis_high, indexing='ij')
            
            # calculate xT values for low and high regimes in large beta
            xT_mesh_low = z_mesh_low * x1_list_lrg[:, None]
            xT_mesh_high = z_mesh_high * xc_list_lrg[:, None]

            # calculate the deflection angle for low and high regimes in large beta
            phi_arr_low = self.get_phi_arr(self.d_phi_dx_lam, beta_mesh_low, xT_mesh_low, is_meshgrid=True)
            phi_arr_high = self.get_phi_arr(self.d_phi_dx_lam, beta_mesh_high, xT_mesh_high, is_meshgrid=True)
            
            # calculate the Jacobian for low and high regimes in large beta
            jac_arr_low = self.jacobian_y_sq_xT_lam(xT_mesh_low, beta_mesh_low)
            jac_arr_high = self.jacobian_y_sq_xT_lam(xT_mesh_high, beta_mesh_high)
            
            # store the results
            # self.beta_arr = np.concatenate([beta_list_small_beta, beta_list_lrg])
            self.jac_arr = {'small_beta': jac_arr_small_beta, 'low': jac_arr_low, 'high': jac_arr_high}
            self.phi_arr = {'small_beta': phi_arr_small_beta, 'low': phi_arr_low, 'high': phi_arr_high}
            self.xT_mesh = {'small_beta': xT_mesh_small_beta, 'low': xT_mesh_low, 'high': xT_mesh_high}
        
        # repulsive case
        else:
            # define z ranges for the repulsive case
            z_lis = np.logspace(0.0001, 2, 200)
            
            # generate meshgrids for z and beta
            beta_mesh, z_mesh = np.meshgrid(self.beta_arr, z_lis, indexing='ij')
            
            # calculate xT values
            xT_mesh = z_mesh * self.xT_min_arr[:, None]


            # calculate the deflection angles on the mesh
            phi_arr = self.get_phi_arr(self.d_phi_dx_lam, beta_mesh, xT_mesh, is_meshgrid=True)
            # calculate the Jacobian on the mesh
            # Initialize jac_arr with 2 * xT_mesh
            jac_arr = 2 * xT_mesh

            # Create a boolean mask for the condition
            mask = xT_mesh + np.log(beta_mesh) <100

            # Calculate the Jacobian only for elements where the condition is True
            jac_arr[mask] = self.jacobian_y_sq_xT_lam(xT_mesh[mask], beta_mesh[mask])

            # store the results
            self.jac_arr = jac_arr
            self.phi_arr = phi_arr
            self.xT_mesh = xT_mesh            
            
 

    def calculate_xsec(self, func='momentum'):
        """
        Calculates sigma values based on the provided function and previously computed data.
        """
        beta_arr, jac_arr, phi_arr, xT_mesh = self.beta_arr, self.jac_arr, self.phi_arr, self.xT_mesh 

        if isinstance(func, str):
            if func not in ['momentum', 'viscosity']:
                raise ValueError('Invalid function name.')
            if func == 'momentum':
                func = lambda x: 1 - np.cos(x)
            elif func == 'viscosity':
                func = lambda x: 1 - np.cos(x)**2

        
        # Attractive case
        if self.is_attractive:
            ds = {}
            for regime in ['small_beta', 'low', 'high']:
                ds[regime] = func(np.pi - 2 * phi_arr[regime]) * np.pi * jac_arr[regime]

            sig_func = {}
            for regime in ['small_beta', 'low', 'high']:
                sig_func[regime] = np.trapz(ds[regime], xT_mesh[regime], axis=1)

            sig_func['large_beta'] = sig_func['low'] + sig_func['high']

            sig_func_arr = np.concatenate([sig_func['small_beta'], sig_func['large_beta']])
        
        # Repulsive case
        
        else:
            ds = func(np.pi - 2 * phi_arr) * np.pi * jac_arr
            sig_func_arr = np.trapz(ds, xT_mesh, axis=1)


        return beta_arr, sig_func_arr

