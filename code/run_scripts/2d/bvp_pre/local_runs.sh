mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=128 --nz=32 --root_dir=runs/bvp_pre --run_time_therm=15 --mixed_flux_T --no_slip --Rayleigh=2.79e3   --do_bvp --bvp_final_equil_time=50  
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=128 --nz=32 --root_dir=runs/bvp_pre --run_time_therm=15 --mixed_flux_T --no_slip --Rayleigh=6.01e3   --do_bvp --bvp_final_equil_time=50 
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=128 --nz=32 --root_dir=runs/bvp_pre --run_time_therm=15 --mixed_flux_T --no_slip --Rayleigh=1.30e4   --do_bvp --bvp_final_equil_time=50 
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=128 --nz=32 --root_dir=runs/bvp_pre --run_time_therm=15 --mixed_flux_T --no_slip --Rayleigh=2.79e4   --do_bvp --bvp_final_equil_time=50 
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=128 --nz=32 --root_dir=runs/bvp_pre --run_time_therm=15 --mixed_flux_T --no_slip --Rayleigh=6.01e4   --do_bvp --bvp_final_equil_time=50  --seed=200
