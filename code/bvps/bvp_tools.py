from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


from mpi4py import MPI
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from dedalus import public as de

class BVPSolverBase:
    """
    A base class for solving a BVP in the middle of a running IVP.

    This class sets up basic functionality for tracking profiles and solving BVPs.
    This is just an abstract class, and must be inherited with specific equation
    sets to work.

    Objects of this class are paired with a dedalus solver which is timestepping forward
    through an IVP.  This class calculates horizontal and time averages of important flow
    fields from that IVP, then uses those as NCCs in a BVP to get a more evolved thermal state

    CLASS VARIABLES
    ---------------
        FIELDS     - An OrderedDict of strings of which the time- and horizontally- averaged 
                     profiles are tracked (and fed into the BVP)
        VARS       - An OrderedDict of variables which will be updated by the BVP
        VEL_VARS   - An OrderedDict of variables which contain info about the velocity field;
                        may be updated by BVP.
        LOCAL_TRACK -An OrderedDict of variables which contain info about the instantaneous horizontal average of a profile

    Object Attributes:
    ------------------
        avg_started         - If True, time averages for FIELDS has begun
        avg_time_elapsed    - Amount of IVP simulation time over which averages have been taken so far
        avg_time_start      - Simulation time at which average began
        bvp_equil_time      - Amount of sim time to wait for velocities to converge before starting averages
                                at the beginning of IVP or after a BVP is solved
        bvp_transient_time  - Amount of time to wait at the beginning of the sim
        bvp_run_threshold   - Degree of convergence required on time averages before doing BVP
        bvp_l2_check_time   - How often to check for convergence, in simulation time
        bvp_l2_last_check_time - Last time we checked if avgs were converged
        current_avg_profiles- A 2 x nz_per_proc array containing the previous and current values of each tracked profile, for use in averaging
        current_avg_times   - A 2 element array containing the time of the previous avg and the time of the current avg.
        current_local_avg   - Current value of the local portion of the time average of profiles
        current_local_l2    - The avg of the abs() of the change in the avg profile compared to the previous timestep.
        comm                - COMM_WORLD for IVP
        completed_bvps      - # of BVPs that have been completed during this run
        do_bvp              - If True, average profiles are converged, do BVP.
        files_saved         - The number of BVP files that have been saved
        final_equil_time    - How long to allow the solution to equilibrate after the final bvp
        first_l2            - If True, we haven't taken an L2 average for convergence yet.
        flow                - A dedalus flow_tools.GlobalFlowProperty object for the IVP solver which is tracking
                                the Reynolds number, and will track FIELDS variables
        mesh                - processor mesh of the IVP in [ny, nz] form
        min_bvp_time        - Minimum simulation time to wait between BVPs.
        min_avg_dt          - Minimum simulation time to wait between adjusting average profiles.
        nz_per_proc          - Number of z-points per core (for parallelization)
        num_bvps            - Total (max) number of BVPs to complete
        nx                  - x-resolution of the IVP grid
        ny                  - y-resolution of the IVP grid
        nz                  - z-resolution of the IVP grid
        nz_per_proc         - number of z-points on each processor (important to track in 3D)
        partial_prof_dict   - a dictionary containing local contributions to the averages of FIELDS
        plot_dir            - A directory to save plots into during BVPs.
        profiles_dict       - a dictionary containing the time/horizontal average of FIELDS
        profiles_dict_curr  - a dictionary containing the time/horizontal average of LOCAL_TRACK for current atmosphere state
        rank                - comm rank
        re_switch           - A reynolds-number-based switch that becomes True once the Re threshold for averaging is reached.
        size                - comm size
        solver              - The corresponding dedalus IVP solver object
        solver_states       - The states of VARS in solver

    """
    
    FIELDS = None
    VARS   = None
    VEL_VARS   = None
    LOCAL_TRACK = None

    def __init__(self, nx, ny, nz, flow, comm, solver, num_bvps, bvp_equil_time, bvp_transient_time=20,
                 bvp_run_threshold=1e-2, first_run_threshold=1e-2, bvp_l2_check_time=1, min_bvp_time=35, first_bvp_time=20, plot_dir=None,
                 min_avg_dt=0.05, final_equil_time = None, mesh=None):
        """
        Initializes the object; grabs solver states and makes room for profile averages
        
        Arguments:
        nx                  - the horizontal resolution of the IVP
        nz                  - the vertical resolution of the IVP
        flow                - a dedalus.extras.flow_tools.GlobalFlowProperty for the IVP solver
        comm                - An MPI comm object for the IVP solver
        solver              - The IVP solver
        min_bvp_time        - Minimum sim time to do average over before doing a bvp
        min_avg_dt          - Minimum sim time to wait between adjusting averages
        num_bvps            - Maximum number of BVPs to solve
        bvp_equil_time      - Sim time to wait after a bvp before beginning averages for the next one
        bvp_transient_time  - Sim time to wait at beginning of simulation before starting average
        bvp_run_threshold   - Level of convergence that must be reached in statistical averages
                                before doing a BVP (1e-2 = 1% variation OK, 1e-3 = 0.1%, so on)
        bvp_l2_check_time   - Sim time to wait between communications to see if we're converged
                                (so that we don't "check for convergence" on all processes at every timestep)
        plot_dir            - If not None, save plots to this directory during bvps.
        """
        #Get info about IVP
        self.flow       = flow
        self.solver     = solver
        self.nx         = nx
        self.ny         = ny
        self.nz         = nz
        self.mesh       = mesh

        #Specify how BVPs work
        self.num_bvps           = num_bvps
        self.min_bvp_time       = min_bvp_time
        self.first_bvp_time     = first_bvp_time
        self.min_avg_dt         = min_avg_dt
        self.completed_bvps     = 0
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = 0.
        self.bvp_equil_time     = bvp_equil_time
        self.bvp_transient_time = bvp_transient_time
        self.avg_started        = False
        self.final_equil_time   = final_equil_time

        # Stop parameters for bvps
        self.bvp_run_threshold      = bvp_run_threshold
        self.first_run_threshold    = first_run_threshold
        self.bvp_l2_check_time      = 1
        self.bvp_l2_last_check_time = 0
        self.do_bvp                 = False
        self.first_l2               = True
        self.re_switch              = False

        #Get info about MPI distribution
        self.comm           = comm
        self.rank           = comm.rank
        self.size           = comm.size
        if self.mesh == None:
            self.nz_per_proc     = self.nz/self.size
        else:
            self.nz_per_proc     = self.nz/self.mesh[-1]

        # Set up tracking dictionaries for flow fields
        for fd, field in self.FIELDS.items():
            self.flow.add_property('plane_avg({})'.format(field), name='{}'.format(fd))
        for fd, field in self.LOCAL_TRACK.items():
            self.flow.add_property('plane_avg({})'.format(field), name='{}'.format(fd))
 
        if self.rank == 0:
            self.profiles_dict = OrderedDict()
            for fd, info in self.FIELDS.items():
                self.profiles_dict[fd]      = np.zeros(nz)

        # Set up a dictionary of partial profiles to track averages locally so we
        # don't have to communicate each timestep.
        self.profiles_dict_curr = OrderedDict()
        self.current_avg_profiles = OrderedDict()
        self.current_local_avg = OrderedDict()
        self.current_local_l2  = OrderedDict()
        for fd, info in self.FIELDS.items():
            self.current_avg_profiles[fd]   = np.zeros((2,self.nz_per_proc))

            self.current_local_avg[fd]  = np.zeros(self.nz_per_proc)
            self.current_local_l2[fd]   = np.zeros(self.nz_per_proc)
        for fd in self.LOCAL_TRACK.keys():
            self.profiles_dict_curr[fd] = np.zeros(self.nz_per_proc)


        self.current_avg_times      = np.zeros(2)
        self.current_avg_times[1]   = self.solver.sim_time

        # Set up a dictionary which tracks the states of important variables in the solver.
        self.solver_states = OrderedDict()
        self.vel_solver_states = OrderedDict()
        for st, var in self.VARS.items():
            self.solver_states[st] = self.solver.state[var]
        for st, var in self.VEL_VARS.items():
            self.vel_solver_states[st] = self.solver.state[var]

        self.plot_dir = plot_dir
        self.files_saved = 0
        if not isinstance(self.plot_dir, type(None)):
            import os
            if self.rank == 0 and not os.path.exists('{:s}'.format(self.plot_dir)):
                os.mkdir('{:s}'.format(self.plot_dir))


    def get_local_profile(self, prof_name):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, 
        update the average on the local core based on the current flows.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
        """
        field = self.flow.properties['{}'.format(prof_name)]['g']
        if len(field.shape) == 3:
            profile = field[0,0,:]
        else:
            profile = field[0,:]
        return profile

    def _update_profiles_dict(self, *args, **kwargs):
        pass

    def get_full_profile(self, dictionary, prof_name):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, communicate the
        full vertical profile across all processes, then return the full profile as a function
        of depth.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
            avg_type        - If 0, horiz avg.  If 1, full 2D field.
        """
        local = np.zeros(self.nz)
        glob  = np.zeros(self.nz)
        if isinstance(self.mesh, type(None)):
            local[self.nz_per_proc*self.rank:self.nz_per_proc*(self.rank+1)] = \
                    dictionary[prof_name]
        elif self.rank < self.mesh[-1]:
            local[self.nz_per_proc*self.rank:self.nz_per_proc*(self.rank+1)] = \
                    dictionary[prof_name]
        self.comm.Allreduce(local, glob, op=MPI.SUM)

        profile = glob
        return profile

    def update_avgs(self, dt, Re_avg, min_Re = 1):
        """
        If proper conditions are met, this function adds the time-weighted vertical profile
        of all profiles in FIELDS to the appropriate arrays which are tracking classes. The
        size of the timestep is also recorded.

        The averages taken by this class are time-weighted averages of horizontal averages, such
        that sum(dt * Profile) / sum(dt) = <time averaged profile used for BVP>

        Arguments:
            dt          - The size of the current timestep taken.
            min_Re      - Only count this timestep toward the average if vol_avg(Re) is greater than this.
        """
        solver_sim_time = self.solver.sim_time
        #Don't average if all BVPs are done
        if self.completed_bvps >= self.num_bvps:
            return

        # Only avg if we're above the Re threshold
        if not self.re_switch:
            if Re_avg > min_Re:
                self.re_switch = True

        if self.re_switch:
            if not self.avg_started: #set flags
                self.avg_started=True
                self.avg_time_start = solver_sim_time

            # Don't count point if a BVP has been completed very recently
            if self.completed_bvps == 0:
                if (solver_sim_time - self.avg_time_start) < self.bvp_transient_time:
                    return
            else:
                if (solver_sim_time - self.avg_time_start) < self.bvp_equil_time:
                    return

            # Grab local profile info
            for fd, info in self.FIELDS.items():
                self.current_avg_profiles[fd][0,:] = self.get_local_profile(fd)
            for fd in self.LOCAL_TRACK.keys():
                self.profiles_dict_curr[fd] = self.get_local_profile(fd)
            
            self.current_avg_times[0] = solver_sim_time
            time_diff = self.current_avg_times[0] - self.current_avg_times[1]
            #Update sums for averages. Check to see if we're converged enough for a BVP.
            if time_diff >= self.min_avg_dt:
                if self.first_l2:
                    self.current_avg_times[1] = self.current_avg_times[0]
                else:
                    self.avg_time_elapsed += time_diff
                for fd, info in self.FIELDS.items():
                    if self.first_l2:
                        self.current_avg_profiles[fd][1,:] = self.current_avg_profiles[fd][0,:]
                        self.current_local_l2[fd]  *= 0
                        continue
                    else:
                        avg = self.current_local_avg[fd]*1. / (self.avg_time_elapsed - time_diff)
                        self.current_local_avg[fd] += ((time_diff)/2)*(self.current_avg_profiles[fd][1,:] + self.current_avg_profiles[fd][0,:])
                        new_avg = self.current_local_avg[fd]*1. / self.avg_time_elapsed
                        self.current_local_l2[fd] = np.abs((new_avg - avg)/new_avg)

                    #get staged for next point in avg.
                    self.current_avg_times[1] = self.current_avg_times[0]
                    self.current_avg_profiles[fd][1,:] = self.current_avg_profiles[fd][0,:]

                # Check if converged for BVP
                if (solver_sim_time - self.bvp_l2_last_check_time) > self.bvp_l2_check_time and not self.first_l2:
                    local, globl = np.zeros(len(self.FIELDS.keys())), np.zeros(len(self.FIELDS.keys()))
                    for i, k in enumerate(self.FIELDS.keys()):
                        local[i] = np.max(self.current_local_l2[k])
                    self.comm.Allreduce(local, globl, op=MPI.MAX)

                    self.do_bvp = False
                    if self.completed_bvps == 0:
                        logger.info('MAX ABS DIFFERENCE IN L2 NORM FOR CONVERGENCE: {:.4g} / {:.4g} FOR BVP SOLVE'.format(np.max(globl), self.first_run_threshold))
                        if np.max(globl) < self.first_run_threshold:
                            self.do_bvp = True
                    else:
                        logger.info('MAX ABS DIFFERENCE IN L2 NORM FOR CONVERGENCE: {:.4g} / {:.4g} FOR BVP SOLVE'.format(np.max(globl), self.bvp_run_threshold))
                        if np.max(globl) < self.bvp_run_threshold:
                            self.do_bvp = True
                    self.bvp_l2_last_check_time = solver_sim_time
                self.first_l2 = False



    def check_if_solve(self):
        """ Returns a boolean.  If True, it's time to solve a BVP """
#        logger.debug('start bvp {}'.format((self.avg_started and self.avg_time_elapsed >= self.min_bvp_time) and (self.do_bvp and (self.completed_bvps < self.num_bvps))))
        if self.completed_bvps == 0:
            term1 = self.avg_time_elapsed >= self.first_bvp_time
        else:
            term1 = self.avg_time_elapsed >= self.min_bvp_time
        term2 = self.completed_bvps < self.num_bvps
        return self.avg_started and term1 and term2 and self.do_bvp

    def _save_file(self):
        """  Saves profiles dict to file """
        if not isinstance(self.plot_dir, type(None)):
            z_profile = np.zeros(self.nz)
            if isinstance(self.mesh, type(None)):
                z_profile[self.rank*self.nz_per_proc:(self.rank+1)*self.nz_per_proc] = self.solver.domain.grid(-1)
            else:
                if self.rank < self.mesh[-1]:
                    z_profile[self.rank*self.nz_per_proc:(self.rank+1)*self.nz_per_proc] = self.solver.domain.grid(-1)
            global_z = np.zeros_like(z_profile)
            self.comm.Allreduce(z_profile, global_z, op=MPI.SUM)
            if self.rank == 0:
                file_name = self.plot_dir + "profile_dict_file_{:04d}.h5".format(self.files_saved)
                with h5py.File(file_name, 'w') as f:
                    for k, item in self.profiles_dict.items():
                        f[k] = item
                    f['z'] = global_z
            self.files_saved += 1
                    
            

    def terminate_IVP(self):
        if not isinstance(self.final_equil_time, type(None)):
            if ((self.solver.sim_time - self.avg_time_start) >= self.final_equil_time) and (self.completed_bvps >= self.num_bvps):
                return True
        else:
            return False
            

    def _reset_fields(self):
        """ Reset all local fields after doing a BVP """
        self.do_bvp = False
        self.first_l2 = True
        # Reset profile arrays for getting the next bvp average
        for fd, info in self.FIELDS.items():
            if self.rank == 0:
                self.profiles_dict[fd]      *= 0
            self.current_local_avg[fd]  *= 0
            self.current_local_l2[fd]  *= 0
            self.current_avg_times *= 0

    def _set_subs(self, problem):
        pass
    
    def _set_eqns(self, problem):
        pass

    def _set_BCs(self, problem):
        pass


    def solve_BVP(self):
        """ Base functionality at the beginning of BVP solves, regardless of equation set"""

        for fd, item in self.FIELDS.items():
            curr_profile = self.get_full_profile(self.current_local_avg, fd)
            if self.rank == 0:
                self.profiles_dict[fd] = curr_profile / self.avg_time_elapsed
        self._save_file()

        # Restart counters for next BVP
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = self.solver.sim_time
        self.completed_bvps     += 1

class BoussinesqBVPSolver(BVPSolverBase):
    """
    Inherits the functionality of BVP_solver_base in order to solve BVPs involving
    the Boussinesq equations in the middle of time evolution of IVPs.

    Solves energy equation.  Makes no approximations other than time-stationary dynamics.
    """

    # 0 - full avg profile
    # 1 - stdev profile
    FIELDS = OrderedDict([  
                ('enth_flux_IVP',       'w*(T0+T1)'),                      
                ('tot_flux_IVP',        '(w*(T0+T1) - P*(T0_z+T1_z))'),                      
                ('momentum_rhs_z',      '(u*Oy - v*Ox)'),                      
                        ])

    LOCAL_TRACK = OrderedDict([  
                ('T1_IVP',              'T1'),                      
                ('p_IVP',               'p'),                      
                        ])


    VARS   = OrderedDict([  
                ('T1_IVP',              'T1'),
                ('T1_z_IVP',            'T1_z'), 
                ('p_IVP',               'p'), 
                        ])
    VEL_VARS_2D = OrderedDict([
                ('u_IVP',               'u'), 
                ('w_IVP',               'w'), 
                ('Oy_IVP',              'Oy'), 
                        ])

    VEL_VARS_3D = OrderedDict([
                ('u_IVP',               'u'), 
                ('v_IVP',               'v'), 
                ('w_IVP',               'w'), 
                ('Ox_IVP',              'Ox'), 
                ('Oy_IVP',              'Oy'), 
                ('Oz_IVP',              'Oz'), 
                        ])

    def __init__(self, atmosphere_class, *args, threeD=False, **kwargs):
        self.atmosphere_class = atmosphere_class
        self.plot_count = 0
        self.threeD = threeD
        if self.threeD:
            self.VEL_VARS = self.VEL_VARS_3D
        else:
            self.VEL_VARS = self.VEL_VARS_2D
        super(BoussinesqBVPSolver, self).__init__(*args, **kwargs)
    
    def _set_eqns(self, problem):
        """ Sets the horizontally-averaged boussinesq equations """
        logger.debug('setting T1_z eqn')
        problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug('Setting energy equation')
        problem.add_equation(("P*dz(T1_z) = dz(enth_flux_IVP - P*T0_z)"))
        
        logger.debug('Setting HS equation')
        problem.add_equation(("dz(p1) - T1 = momentum_rhs_z"))
        
    def _set_BCs(self, atmosphere, bc_kwargs):
        """ Sets standard thermal BCs, and also enforces the m = 0 pressure constraint """
        atmosphere.dirichlet_set = []
        if bc_kwargs['fixed_flux']:
            logger.info("Thermal BC: fixed_flux (BVP form)")
            atmosphere.problem.add_bc( "left(T1_z) = 0")
            atmosphere.problem.add_bc( "left(T1) + right(T1) = 0")
            atmosphere.dirichlet_set.append('T1')
            atmosphere.dirichlet_set.append('T1_z')
        else:
            atmosphere.set_thermal_BC(**bc_kwargs)
        atmosphere.problem.add_bc('right(p1) = 0')
        for key in atmosphere.dirichlet_set:
            atmosphere.problem.meta[key]['z']['dirichlet'] = True

    def _update_profiles_dict(self, bc_kwargs, atmosphere, vel_adjust_factor):
        """
        Update the enthalpy flux profile such that the BVP solve gets us the right answer.
        """

        #Get the atmospheric z-points (on the right size grid)
        z = atmosphere._new_field()
        z['g'] = atmosphere.z
        z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        z = z['g']

        #Keep track of initial flux profiles, then make a new enthalpy flux profile
        init_kappa_flux = 1*self.profiles_dict['tot_flux_IVP'] - self.profiles_dict['enth_flux_IVP']
        init_enth_flux = 1*self.profiles_dict['enth_flux_IVP']

        atmosphere.T0_z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        flux_through_system = -atmosphere.P * atmosphere.T0_z['g']
        flux_scaling = flux_through_system / self.profiles_dict['tot_flux_IVP']

        #Scale flux appropriately.
        self.profiles_dict['enth_flux_IVP'] *= flux_scaling #flux_through_system/self.profiles_dict['tot_flux_IVP']
        self.profiles_dict['momentum_rhs_z'] *= flux_scaling #flux_through_system/self.profiles_dict['tot_flux_IVP']

        #Make some plots
        if not isinstance(self.plot_dir, type(None)):
            plt.plot(z, init_kappa_flux + init_enth_flux)
            plt.plot(z, init_kappa_flux)
            plt.plot(z, init_enth_flux)
            plt.plot(z, self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.plot(z, flux_through_system)
            plt.savefig('{}/fluxes_before_{:04d}.png'.format(self.plot_dir, self.plot_count))
            plt.close()
            print('flux ratio', self.profiles_dict['enth_flux_IVP']/init_enth_flux)
            plt.plot(z, self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.plot(z, flux_through_system - self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.savefig('{}/fluxes_after_{:04d}.png'.format(self.plot_dir, self.plot_count))
            plt.close()
            for fd in self.FIELDS.keys():
                plt.plot(z, self.profiles_dict[fd])
                plt.savefig('{}/{}_{:04d}.png'.format(self.plot_dir, fd, self.plot_count))
                plt.close()
        self.plot_count += 1

        return flux_scaling


    def solve_BVP(self, atmosphere_kwargs, diffusivity_args, bc_kwargs, tolerance=1e-10):
        """
        Solves a BVP in a 2D Boussinesq box.

        The BVP calculates updated temperature / pressure fields, then updates 
        the solver states which are tracked in self.solver_states.  
        This automatically updates the IVP's fields.

        """
        super(BoussinesqBVPSolver, self).solve_BVP()
        nz = atmosphere_kwargs['nz']
        # Create space for the returned profiles on all processes.
        return_dict = OrderedDict()
        for v in self.VARS.keys():
            return_dict[v] = np.zeros(self.nz, dtype=np.float64)
        return_dict['flux_scaling'] = np.zeros(self.nz, dtype=np.float64)
        return_dict['enth_flux_IVP'] = np.zeros(self.nz, dtype=np.float64)


        # No need to waste processor power on multiple bvps, only do it on one
        if self.rank == 0:
            vel_adjust_factor = 1
            atmosphere = self.atmosphere_class(dimensions=1, comm=MPI.COMM_SELF, **atmosphere_kwargs)
            atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z','p1'], ncc_cutoff=tolerance)

            #Zero out old varables to make atmospheric substitutions happy.
            if self.threeD:
                old_vars = ['u', 'v', 'w', 'dx(A)', 'dy(A)', 'Ox', 'Oy', 'Oz']
            else:
                old_vars = ['u', 'w', 'dx(A)', 'Oy']
            for sub in old_vars:
                atmosphere.problem.substitutions[sub] = '0'

            atmosphere._set_parameters(*diffusivity_args)
            atmosphere._set_subs()

            # Create the appropriate enthalpy flux profile based on boundary conditions
            return_dict['flux_scaling'] = self._update_profiles_dict(bc_kwargs, atmosphere, vel_adjust_factor)
            return_dict['enth_flux_IVP'] = self.profiles_dict['enth_flux_IVP']
            f = atmosphere._new_ncc()
            f.set_scales(self.nz / nz, keep_data=True) #If nz(bvp) =/= nz(ivp), this allows interaction between them
            f['g'] = return_dict['flux_scaling']
            vel_adjust_factor = np.sqrt(np.mean(f.integrate('z')['g'])/atmosphere.Lz)

            #Add time and horizontally averaged profiles from IVP to the problem as parameters
            for k in self.FIELDS.keys():
                f = atmosphere._new_ncc()
                f.set_scales(self.nz / nz, keep_data=True) #If nz(bvp) =/= nz(ivp), this allows interaction between them
                if len(self.profiles_dict[k].shape) == 2:
                    f['g'] = self.profiles_dict[k].mean(axis=0)
                else:
                    f['g'] = self.profiles_dict[k]
                atmosphere.problem.parameters[k] = f

            self._set_eqns(atmosphere.problem)
            self._set_BCs(atmosphere, bc_kwargs)

            # Solve the BVP
            solver = atmosphere.problem.build_solver()

            pert = solver.perturbations.data
            pert.fill(1+tolerance)
            while np.sum(np.abs(pert)) > tolerance:
                solver.newton_iteration()
                logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))


            T1 = solver.state['T1']
            P1   = solver.state['p1']

            #Appropriately adjust T1 in IVP
            T1.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] += T1['g']

            #Appropriately adjust p in IVP
            P1.set_scales(self.nz/nz, keep_data=True)
            return_dict['p_IVP'] += P1['g']
        else:
            for v in self.VARS.keys():
                return_dict[v] *= 0
        logger.info(return_dict)
        self.comm.Barrier()
        # Communicate output profiles from proc 0 to all others.
        for v in self.VARS.keys():
            glob = np.zeros(self.nz)
            self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
            return_dict[v] = glob
            glob = np.zeros(self.nz)

        for v in ['flux_scaling', 'enth_flux_IVP']:
            glob = np.zeros(self.nz)
            self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
            return_dict[v] = glob

        vel_adj_loc = np.zeros(1)
        vel_adj_glob = np.zeros(1)
        if self.rank == 0:
            vel_adj_loc[0] = vel_adjust_factor
        self.comm.Allreduce(vel_adj_loc, vel_adj_glob, op=MPI.SUM)
        
        if isinstance(self.mesh, type(None)):
            z_rank = self.rank
        else:
            z_rank = self.rank % self.mesh[-1]

        local_flux_reducer =  return_dict['flux_scaling'][self.nz_per_proc*z_rank:self.nz_per_proc*(z_rank+1)]
        local_flux_reducer = np.sqrt(local_flux_reducer)


        logger.info('updating thermo states')
        # Actually update IVP states
        for v in self.VARS.keys():
            if v == 'T1_z_IVP':
                continue
            #Subtract out current avg
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] -= self.profiles_dict_curr[v]

            if v == 'T1_IVP': #modify the fluctuating parts of T1
                self.solver_states[v].set_scales(1, keep_data=True)
                self.solver_states[v]['g'] *= local_flux_reducer
            #Put in right avg
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += return_dict[v][self.nz_per_proc*z_rank:self.nz_per_proc*(z_rank+1)]

        self.solver_states['T1_IVP'].differentiate('z', out=self.solver_states['T1_z_IVP'])

        logger.info('updating velocity states')
        for v in self.VEL_VARS.keys():
            self.vel_solver_states[v].set_scales(1, keep_data=True)
            self.vel_solver_states[v]['g'] *= local_flux_reducer
        self._reset_fields()

