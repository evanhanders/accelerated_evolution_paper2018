"""
    This file is a partial driving script for boussinesq dynamics.  Here,
    formulations of the boussinesq equations are handled in a clean way using
    classes.
"""
import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
    """
    A general, abstract class for solving equations in dedalus.

    This class can be inherited by other classes to set up specific equation sets, but
    the base (parent) class contains much of the logic we will need regardless (setting
    up the domain, creating a problem or a new non-constant coefficient, etc.)

    Attributes:
        compound        - If True, z-basis is a set of compound chebyshevs
        dimensions      - The dimensionality of the problem (1D, 2D, 3D)
        domain          - The dedalus domain on which the problem is being solved
        mesh            - The processor mesh over which the problem is being solved
        problem         - The Dedalus problem object that is being solved
        problem_type    - The type of problem being solved (IVP, EVP)
        x, y, z         - 1D NumPy arrays containing the physical coordinates of grid points in grid space
        Lx, Ly, Lz      - Scalar containing the size of the atmosphere in x, y, z directions
        nx, ny, nz      - Scalars containing the number of points in the x, y, z directions
        delta_x, delta_y- Grid spacings in the x, y directions (assuming constant grid spacing)
        z_dealias       - 1D NumPy array containing the dealiased locations of grid points in the z-direction
    """

    def __init__(self, dimensions=2):
        """Initialize all object attributes"""
        self.compound       = False
        self.dimensions     = dimensions
        self.domain         = None
        self.mesh           = None
        self.problem        = None
        self.problem_type   = ''
        self.x, self.Lx, self.nx, self.delta_x   = [None]*4
        self.y, self.Ly, self.ny, self.delta_y   = [None]*4
        self.z, self.z_dealias, self.Lz, self.nz = [None]*4
        return

    def set_domain(self, nx=256, Lx=4,
                          ny=256, Ly=4,
                          nz=128, Lz=1,
                          grid_dtype=np.float64, comm=MPI.COMM_WORLD, mesh=None):
        """
        Here the dedalus domain is created for the equation set

        Inputs:
            nx, ny, nz      - Number of grid points in the x, y, z directions
            Lx, Ly, Lz      - Physical size of the x, y, z direction
            grid_dtype      - Datatype to use for grid points in the problem
            comm            - Comm group over which to solve.  Use COMM_SELF for EVP
            mesh            - The processor mesh over which the problem is solved.
        """
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        self.mesh=mesh
        
        if not isinstance(nz, list):
            nz = [nz]
        if not isinstance(Lz, list):
            Lz = [Lz]   

        if len(nz)>1:
            logger.info("Setting compound basis in vertical (z) direction")
            z_basis_list = []
            Lz_interface = 0.
            for iz, nz_i in enumerate(nz):
                Lz_top = Lz[iz]+Lz_interface
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=3/2)
                z_basis_list.append(z_basis)
                Lz_interface = Lz_top
            self.compound = True
            z_basis = de.Compound('z', tuple(z_basis_list),  dealias=3/2)
        elif len(nz)==1:
            logger.info("Setting single chebyshev basis in vertical (z) direction")
            z_basis = de.Chebyshev('z', nz[0], interval=[0, Lz[0]], dealias=3/2)
        
        if self.dimensions > 1:
            x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        if self.dimensions > 2:
            y_basis = de.Fourier(  'y', ny, interval=[0., Ly], dealias=3/2)
        if self.dimensions == 1:
            bases = [z_basis]
        elif self.dimensions == 2:
            bases = [x_basis, z_basis]
        elif self.dimensions == 3:
            bases = [x_basis, y_basis, z_basis]
        else:
            logger.error('>3 dimensions not implemented')
        
        self.domain = de.Domain(bases, grid_dtype=grid_dtype, comm=comm, mesh=mesh)
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)

        if self.dimensions == 1:
            self.Lx, self.Ly = 0, 0
        if self.dimensions > 1:
            self.x = self.domain.grid(0)
            self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.nx = self.domain.bases[0].coeff_size
            self.delta_x = self.Lx/self.nx
        if self.dimensions > 2:
            self.y = self.domain.grid(1)
            self.Ly = self.domain.bases[1].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.ny = self.domain.bases[1].coeff_size
            self.delta_y = self.Ly/self.ny
    
    def set_IVP(self, *args, ncc_cutoff=1e-10, **kwargs):
        """
        Constructs and initial value problem of the current object's equation set
        """
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_EVP(self, *args, ncc_cutoff=1e-10, tolerance=1e-10, **kwargs):
        """
        Constructs an eigenvalue problem of the current objeect's equation set.
        Note that dt(f) = omega * f, not i * omega * f, so real parts of omega
        are growth / shrinking nodes, imaginary parts are oscillating.
        """

        self.problem_type = 'EVP'
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=tolerance)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def set_equations(self, *args, **kwargs):
        """ This function must be implemented in child objects of this class """
        pass

    def get_problem(self):
        return self.problem

    def _new_ncc(self):
        """
        Create a new field of the atmosphere from the dedalus domain. Field's metadata is
        set so that it is constant in the x- and y- directions (but can vary in the z).
        """
        # is this used at all in equations.py (other than rxn), or just in atmospheres?
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        # should "necessary quantities" logic occur here?
        field = self.domain.new_field()
        if self.dimensions > 1:
            field.meta['x']['constant'] = True
        if self.dimensions > 2:
            field.meta['y']['constant'] = True            
        return field

    def _new_field(self):
        """Create a new field of the atmosphere that is NOT a NCC. """
        field = self.domain.new_field()
        return field

    def _set_subs(self):
        """ This function must be implemented in child objects of this class """
        pass

    def global_noise(self, seed=42, **kwargs):
        """
        Create a field fielled with random noise of order 1.  Modify seed to
        get varying noise, keep seed the same to directly compare runs.
        """
        # Random perturbations, initialized globally for same results in parallel
        gshape = self.domain.dist.grid_layout.global_shape(scales=self.domain.dealias)
        slices = self.domain.dist.grid_layout.slices(scales=self.domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field = self._new_field()
        noise_field.set_scales(self.domain.dealias, keep_data=False)
        noise_field['g'] = noise
        self.filter_field(noise_field, **kwargs)

        return noise_field

    def filter_field(self, field, frac=0.25):
        """
        Filter a field in coefficient space by cutting off all coefficient above
        a given threshold.  This is accomplished by changing the scale of a field,
        forcing it into coefficient space at that small scale, then coming back to
        the original scale.

        Inputs:
            field   - The dedalus field to filter
            frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                        The upper 75% of coefficients are set to 0.
        """
        dom = field.domain
        logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
        orig_scale = field.meta[:]['scale']
        field.set_scales(frac, keep_data=True)
        field['c']
        field['g']
        field.set_scales(orig_scale, keep_data=True)


class BoussinesqEquations(Equations):
    """
    An extension of the Equations class which contains the full 2D form of the boussinesq
    equations.   
    """
    def __init__(self, *args,  dimensions=2, **kwargs):
        """ 
            Set up class and dedalus domain
        """
        super(BoussinesqEquations, self).__init__(dimensions=dimensions)
        self.set_domain(*args, **kwargs)

    def _set_parameters(self, Rayleigh, Prandtl):
        """
        Set up important parameters of the problem for boussinesq convection
        """
        self.T0_z      = self._new_ncc()
        self.T0_z['g'] = -1
        self.T0        = self._new_ncc()
        self.T0['g']   = self.Lz/2 - self.domain.grid(-1)
        self.T0_zz = self._new_ncc()
        self.T0_zz['g'] = 0
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['T0_z'] = self.T0_z
        self.problem.parameters['T0_zz'] = self.T0_zz

        # Characteristic scales (things we've non-dimensionalized on
        self.problem.parameters['t_buoy']   = 1.
        self.problem.parameters['v_ff']     = 1.

        self.problem.parameters['Rayleigh'] = Rayleigh
        self.problem.parameters['Prandtl']  = Prandtl

        self.P = (Rayleigh * Prandtl)**(-1./2)
        self.R = (Rayleigh / Prandtl)**(-1./2)
        self.problem.parameters['P'] = (Rayleigh * Prandtl)**(-1./2)
        self.problem.parameters['R'] = (Rayleigh / Prandtl)**(-1./2)
        self.thermal_time = (Rayleigh / Prandtl)**(1./2)

        self.problem.parameters['Lz'] = self.Lz
       
    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        self.problem.substitutions['Ox'] = '(dy(w) - dz(v))'
        self.problem.substitutions['Oz'] = '(dx(v) - dy(u))'
        #Diffusivities; diffusive timescale
        self.problem.substitutions['chi']= '(v_ff * Lz * P)'
        self.problem.substitutions['nu'] = '(v_ff * Lz * R)'
        self.problem.substitutions['t_therm'] = '(Lz**2/chi)'
        
        self.problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['enstrophy'] = '(Ox**2 + Oy**2 + Oz**2)'

        self.problem.substitutions['u_fluc'] = '(u - plane_avg(u))'
        self.problem.substitutions['w_fluc'] = '(w - plane_avg(w))'
        self.problem.substitutions['KE'] = '(0.5*vel_rms**2)'

        self.problem.substitutions['Re'] = '(vel_rms / nu)'
        self.problem.substitutions['Pe'] = '(vel_rms / chi)'
        
        self.problem.substitutions['enth_flux_z']  = '(w*(T1+T0))'
        self.problem.substitutions['kappa_flux_z'] = '(-P*(T1_z+T0_z))'
        self.problem.substitutions['conv_flux_z']  = '(enth_flux_z + kappa_flux_z)'
        self.problem.substitutions['delta_T']      = '(right(T1 + T0) - left(T1 + T0))' 
        self.problem.substitutions['Nu']           = '(1 + enth_flux_z/(-P*delta_T))'

    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None):
        """
        Sets the velocity and thermal boundary conditions at the upper and lower boundaries.  Choose
        one thermal type of BC and one velocity type of BC to set those conditions.  See
        set_thermal_BC() and set_velocity_BC() functions for default choices and specific formulations.
        """
        self.dirichlet_set = []

        self.set_thermal_BC(fixed_flux=fixed_flux, fixed_temperature=fixed_temperature,
                            mixed_flux_temperature=mixed_flux_temperature, mixed_temperature_flux=mixed_temperature_flux)
        
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)
        
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        """
        Sets the thermal boundary conditions at the top and bottom of the atmosphere.  If no choice is made, then the
        default BC is fixed flux (bottom), fixed temperature (top).

        Choices:
            fixed_flux              - T1_z = 0 at top and bottom
            fixed_temperature       - T1 = 0 at top and bottom
            mixed_flux_temperature  - T1_z = 0 at bottom, T1 = 0 at top
            mixed_temperature_flux  - T1 = 0 at bottom, T1_z = 0 at top.
        """
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True

        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z) = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z) = 0")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        """
        Sets the velocity boundary conditions at the top and bottom of the atmosphere.  If no choice is made, then the
        default BC is no slip (top and bottom)

        Boundaries are, by default, impenetrable (w = 0 at top and bottom)

        Choices:
            stress_free         - Oy = 0 at top and bottom [note: Oy = dz(u) - dx(w). With
                                    impenetrable boundaries at top and bottom, dx(w) = 0, so
                                    really these are dz(u) = 0 boundary conditions]
            no_slip             - u = 0 at top and bottom.
        """

        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc("left(Oy) = 0")
            self.problem.add_bc("right(Oy) = 0")
            self.dirichlet_set.append('Oy')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        if self.dimensions == 2:
            self.problem.add_bc("right(p) = 0", condition="(nx == 0)")
            self.problem.add_bc("right(w) = 0", condition="(nx != 0)")
        elif self.dimensions ==3:
            self.problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
            self.problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")
        else:
            self.problem.add_bc("right(p) = 0")
        self.dirichlet_set.append('w')
        
    def set_IC(self, solver, A0=1e-6, **kwargs):
        """
        Set initial conditions as random noise.  I *think* characteristic
        temperature perturbutations are on the order of P, as in the energy
        equation, so our perturbations should be small, comparably (to start
        at a low Re even at large Ra, this is necessary)
        """
        # initial conditions
        T_IC = solver.state['T1']
        T_z_IC = solver.state['T1_z']
            
        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        T_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T_IC['g'] = A0*self.P*np.sin(np.pi*self.z_dealias/self.Lz)*noise['g']*self.T0['g']
        T_IC.differentiate('z', out=T_z_IC)
        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))

    def initialize_output(self, solver, data_dir, 
                          max_writes=20, output_dt=0.25,
                          mode="overwrite", **kwargs):
        """
        Sets up output from runs.
        """

        # Analysis
        analysis_tasks = []
        profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        profiles.add_task("plane_avg(T1+T0)", name="T")
        profiles.add_task("plane_avg(dz(T1+T0))", name="Tz")
        profiles.add_task("plane_avg(T1)", name="T1")
        profiles.add_task("plane_avg(u)", name="u")
        profiles.add_task("plane_avg(w)", name="w")
        profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
        profiles.add_task("plane_avg(Nu)", name="Nu")
        profiles.add_task("plane_avg(Re)", name="Re")
        profiles.add_task("plane_avg(Pe)", name="Pe")
        profiles.add_task("plane_avg(enth_flux_z)", name="enth_flux")
        profiles.add_task("plane_avg(kappa_flux_z)", name="kappa_flux")
        profiles.add_task("plane_avg(kappa_flux_z + enth_flux_z)", name="tot_flux")

        analysis_tasks.append(profiles)

        scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        scalar.add_task("vol_avg(T1)", name="IE")
        scalar.add_task("vol_avg(KE)", name="KE")
        scalar.add_task("vol_avg(T1) + vol_avg(KE)", name="TE")
        scalar.add_task("0.5*vol_avg(u_fluc*u_fluc+w_fluc*w_fluc)", name="KE_fluc")
        scalar.add_task("0.5*vol_avg(u*u)", name="KE_x")
        scalar.add_task("0.5*vol_avg(w*w)", name="KE_z")
        scalar.add_task("0.5*vol_avg(u_fluc*u_fluc)", name="KE_x_fluc")
        scalar.add_task("0.5*vol_avg(w_fluc*w_fluc)", name="KE_z_fluc")
        scalar.add_task("vol_avg(plane_avg(u)**2)", name="u_avg")
        scalar.add_task("vol_avg((u - plane_avg(u))**2)", name="u1")
        scalar.add_task("vol_avg(Nu)", name="Nu")
        scalar.add_task("vol_avg(Re)", name="Re")
        scalar.add_task("vol_avg(Pe)", name="Pe")
        scalar.add_task("vol_avg(delta_T)", name="delta_T")
        analysis_tasks.append(scalar)

        return analysis_tasks


class BoussinesqEquations2D(BoussinesqEquations):

    def __init__(self, *args, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1 - Temperature fluctuations from static state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity
            w    - Vertical velocity
            Oy   - y-vorticity (out of plane)
        """
        super(BoussinesqEquations2D, self).__init__(*args, **kwargs)
        self.variables=['T1_z','T1','p','u','w','Oy']


    def _set_parameters(self, *args):
        """
        Set up important parameters of the problem for boussinesq convection
        """
        super(BoussinesqEquations2D, self)._set_parameters(*args)
        self.problem.parameters['Lx'] = self.Lx

    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        if self.dimensions == 1:
            self.problem.substitutions['plane_avg(A)'] = 'A'
            self.problem.substitutions['plane_std(A)'] = '0'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        else:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

        self.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + w * A_z)'
        self.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dz(A_z))'
       
        self.problem.substitutions['v'] = '0'
        self.problem.substitutions['dy(A)'] = '0'

        super(BoussinesqEquations2D, self)._set_subs()

    def set_equations(self, Rayleigh, Prandtl, kx = 0):
        """
        Set the Boussinesq, Incompressible equations:

            ∇ · u = 0
            d_t u - u ⨯ ω = - ∇ p + T1 (zhat) - √(Pr/Ra) * ∇ ⨯ ω
            d_t T1 + u · ∇ (T0 + T1) = 1/(√[Pr Ra]) * ∇ ² T1

        Here, the form of the momentum equation has been recovered from a more
        familiar form:
            d_t u + u · ∇ u = - ∇ p + T1 (zhat) + √(Pr/Ra) * ∇ ² u,
        where vector operations have been used to express the equation mostly in terms
        of vorticity.  There is a leftover term in
            u · ∇ u = (1/2) ∇ u² - u ⨯ ω,
        but this u² term gets swept into the ∇ p term in boussinesq convection, where p 
        enforces ∇ · u = 0
        """
        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
 
        self._set_parameters(Rayleigh, Prandtl)
        self._set_subs()

        # This formulation is numerically faster to run than the standard form.
        # 2D Boussinesq hydrodynamics

        logger.debug('Adding Eqn: Incompressibility constraint')
        self.problem.add_equation("dx(u) + dz(w) = 0")
        logger.debug('Adding Eqn: T1_z defn')
        self.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: Vorticity defn')
        self.problem.add_equation("Oy - dz(u) + dx(w) = 0")
        logger.debug('Adding Eqn: Momentum, x')
        self.problem.add_equation("dt(u)  - R*dz(Oy)  + dx(p)              =  v*Oz - w*Oy ")
        logger.debug('Adding Eqn: Momentum, z')
        self.problem.add_equation("dt(w)  + R*dx(Oy)  + dz(p)    - T1      =  u*Oy - v*Ox ")
        logger.debug('Adding Eqn: Energy')
        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z   = -UdotGrad(T1, T1_z)")

    def initialize_output(self, solver, data_dir, coeff_output=False,
                          max_writes=20, max_slice_writes=20, output_dt=0.25,
                          mode="overwrite", **kwargs):
        """
        Sets up output from runs.
        """
        
        analysis_tasks = super(BoussinesqEquations2D, self).initialize_output(solver, data_dir, max_writes=max_writes, output_dt=output_dt, mode=mode, **kwargs)

        # Analysis
        snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
        snapshots.add_task("T1 + T0", name='T')
        snapshots.add_task("enstrophy")
        snapshots.add_task("vel_rms")
        snapshots.add_task("u")
        snapshots.add_task("w")
        analysis_tasks.append(snapshots)

        powers = solver.evaluator.add_file_handler(data_dir+'powers', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
        powers.add_task("interp(T1,         z={})".format(self.Lz/2),    name='T midplane', layout='c')
        powers.add_task("interp(T1,         z={})".format(0.05*self.Lz), name='T near bot', layout='c')
        powers.add_task("interp(T1,         z={})".format(0.95*self.Lz), name='T near top', layout='c')
        powers.add_task("interp(u,         z={})".format(self.Lz/2),    name='u midplane' , layout='c')
        powers.add_task("interp(u,         z={})".format(0.05*self.Lz), name='u near bot' , layout='c')
        powers.add_task("interp(u,         z={})".format(0.95*self.Lz), name='u near top' , layout='c')
        powers.add_task("interp(w,         z={})".format(self.Lz/2),    name='w midplane' , layout='c')
        powers.add_task("interp(w,         z={})".format(0.05*self.Lz), name='w near bot' , layout='c')
        powers.add_task("interp(w,         z={})".format(0.95*self.Lz), name='w near top' , layout='c')
        for i in range(10):
            fraction = 0.1*i
            powers.add_task("interp(T1,     x={})".format(fraction*self.Lx), name='T at x=0.{}Lx'.format(i), layout='c')
        analysis_tasks.append(powers)



        if coeff_output:
            coeffs = solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
            coeffs.add_task("T1+T0", name="T", layout='c')
            coeffs.add_task("T1 - plane_avg(T1)", name="T'", layout='c')
            coeffs.add_task("w", layout='c')
            coeffs.add_task("u", layout='c')
            coeffs.add_task("enstrophy", layout='c')
            analysis_tasks.append(coeffs)

        return analysis_tasks

class BoussinesqEquations3D(BoussinesqEquations):

    def __init__(self, *args, dimensions=3, **kwargs):
        """ 
        Initialize class and set up variables that will be used in eqns:
            T1 - Temperature fluctuations from static state
            T1_z - z-derivative of T1
            p    - Pressure, magic
            u    - Horizontal velocity (x)
            u_z   - z-derivative of u
            v    - Horizontal velocity (y)
            v_z  - z-derivative of v
            w    - Vertical velocity
            w_z  - z-derivative of w
        """
        super(BoussinesqEquations3D, self).__init__(*args, dimensions=dimensions, **kwargs)
        self.variables=['T1','T1_z','p','u','v', 'w','u_z', 'v_z', 'w_z']


    def _set_parameters(self, *args):
        """
        Set up important parameters of the problem for boussinesq convection
        """
        super(BoussinesqEquations3D, self)._set_parameters(*args)
        self.problem.parameters['Lx'] = self.Lx
        self.problem.parameters['Ly'] = self.Ly

    def _set_subs(self):
        """
        Sets up substitutions that are useful for the Boussinesq equations or for outputs
        """
        if self.dimensions == 1:
            self.problem.substitutions['plane_avg(A)'] = 'A'
            self.problem.substitutions['plane_std(A)'] = '0'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        else:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
            self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'

        self.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + v * dy(A) + w * A_z)'
        self.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'

        self.problem.substitutions['Oy']          = '(dz(u) - dx(w))'
        self.problem.substitutions['v_fluc'] = '(v - plane_avg(v))'
        super(BoussinesqEquations3D, self)._set_subs()

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        """
        Sets the velocity boundary conditions at the top and bottom of the atmosphere.  If no choice is made, then the
        default BC is no slip (top and bottom)

        Boundaries are, by default, impenetrable (w = 0 at top and bottom)

        Choices:
            stress_free         - Oy = 0 at top and bottom [note: Oy = dz(u) - dx(w). With
                                    impenetrable boundaries at top and bottom, dx(w) = 0, so
                                    really these are dz(u) = 0 boundary conditions]
            no_slip             - u = 0 at top and bottom.
        """

        super(BoussinesqEquations3D, self).set_velocity_BC(stress_free=stress_free, no_slip=no_slip)

        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc("left(Ox) = 0")
            self.problem.add_bc("right(Ox) = 0")
            self.dirichlet_set.append('Ox')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(v) = 0")
            self.problem.add_bc("right(v) = 0")
            self.dirichlet_set.append('v')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

    def set_equations(self, Rayleigh, Prandtl, kx = 0, ky = 0):
        """
        Set the Boussinesq, Incompressible equations:

            ∇ · u = 0
            d_t u + u · ∇ u = - ∇ p + T1 (zhat) + √(Pr/Ra) * ∇ ² u,
            d_t T1 + u · ∇ (T0 + T1) = 1/(√[Pr Ra]) * ∇ ² T1

        """
        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
            self.problem.substitutions['dy(f)'] = "j*ky*(f)"
            self.problem.parameters['ky'] = ky
 
        self._set_parameters(Rayleigh, Prandtl)
        self._set_subs()


        # 3D Boussinesq hydrodynamics
        logger.debug('Adding Eqn: Incompressibility constraint')
        self.problem.add_equation("dx(u) + dy(v) + w_z = 0")
        logger.debug('Adding Eqn: Energy')
        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z   = -UdotGrad(T1, T1_z)")
        logger.debug('Adding Eqn: Momentum, x')
        self.problem.add_equation("dt(u)  - R*Lap(u, u_z) + dx(p)       =  -UdotGrad(u, u_z) ")
        logger.debug('Adding Eqn: Momentum, x')
        self.problem.add_equation("dt(v)  - R*Lap(v, v_z) + dy(p)       =  -UdotGrad(v, v_z) ")
        logger.debug('Adding Eqn: Momentum, z')
        self.problem.add_equation("dt(w)  - R*Lap(w, w_z) + dz(p) - T1  =  -UdotGrad(w, w_z) ")
        logger.debug('Adding Eqn: T1_z defn')
        self.problem.add_equation("T1_z - dz(T1) = 0")
        logger.debug('Adding Eqn: u_z defn')
        self.problem.add_equation("u_z  - dz(u) = 0")
        logger.debug('Adding Eqn: v_z defn')
        self.problem.add_equation("v_z  - dz(v) = 0")
        logger.debug('Adding Eqn: w_z defn')
        self.problem.add_equation("w_z  - dz(w) = 0")

    def initialize_output(self, solver, data_dir, volumes_output=False,
                          max_writes=20, max_slice_writes=20, output_dt=0.25,
                          mode="overwrite", **kwargs):
        """
        Sets up output from runs.
        """
        
        analysis_tasks = super(BoussinesqEquations3D, self).initialize_output(solver, data_dir, max_writes=max_writes, output_dt=output_dt, mode=mode, **kwargs)

        # Analysis
        snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
        snapshots.add_task("interp(T1 + T0,         y={})".format(self.Ly/2), name='T')
        snapshots.add_task("interp(T1 + T0,         z={})".format(0.95*self.Lz), name='T near top')
        snapshots.add_task("interp(T1 + T0,         z={})".format(self.Lz/2), name='T midplane')
        snapshots.add_task("interp(w,         y={})".format(self.Ly/2), name='w')
        snapshots.add_task("interp(w,         z={})".format(0.95*self.Lz), name='w near top')
        snapshots.add_task("interp(w,         z={})".format(self.Lz/2), name='w midplane')
        snapshots.add_task("interp(enstrophy,         y={})".format(self.Ly/2),    name='enstrophy')
        snapshots.add_task("interp(enstrophy,         z={})".format(0.95*self.Lz), name='enstrophy near top')
        snapshots.add_task("interp(enstrophy,         z={})".format(self.Lz/2),    name='enstrophy midplane')
        analysis_tasks.append(snapshots)

        if volumes_output:
            analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
            analysis_volume.add_task("T1 + T0", name="T")
            analysis_volume.add_task("enstrophy", name="enstrophy")
            analysis_tasks.append(analysis_volume)

        return analysis_tasks
