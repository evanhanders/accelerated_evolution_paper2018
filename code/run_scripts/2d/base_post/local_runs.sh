mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=64 --nz=32 --root_dir=runs/base_post --run_time_buoy=100 --mixed_flux_T --no_slip --Rayleigh=2.79e3 --restart=runs/base_pre/rayleigh_benard_2D_mixed_noSlip_Ra2.79e3_Pr1_a2/final_checkpoint/final_checkpoint_s1.h5 --overwrite
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=64 --nz=32 --root_dir=runs/base_post --run_time_buoy=100 --mixed_flux_T --no_slip --Rayleigh=6.01e3 --restart=runs/base_pre/rayleigh_benard_2D_mixed_noSlip_Ra6.01e3_Pr1_a2/final_checkpoint/final_checkpoint_s1.h5 --overwrite
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=64 --nz=32 --root_dir=runs/base_post --run_time_buoy=100 --mixed_flux_T --no_slip --Rayleigh=1.30e4 --restart=runs/base_pre/rayleigh_benard_2D_mixed_noSlip_Ra1.30e4_Pr1_a2/final_checkpoint/final_checkpoint_s1.h5 --overwrite
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=64 --nz=32 --root_dir=runs/base_post --run_time_buoy=100 --mixed_flux_T --no_slip --Rayleigh=2.79e4 --restart=runs/base_pre/rayleigh_benard_2D_mixed_noSlip_Ra2.79e4_Pr1_a2/final_checkpoint/final_checkpoint_s1.h5 --overwrite
mpirun -n 2 python3 rayleigh_benard.py --aspect=2 --nx=64 --nz=32 --root_dir=runs/base_post --run_time_buoy=100 --mixed_flux_T --no_slip --Rayleigh=6.01e4 --restart=runs/base_pre/rayleigh_benard_2D_mixed_noSlip_Ra6.01e4_Pr1_a2/final_checkpoint/final_checkpoint_s1.h5 --overwrite

