
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

python tools\fit_all_frames_mp.py `
   --joints_npy out\miura_joints21.npy `
   --side right `
   --start 0 --end 200 `
   --workers 8 `
   --only_final `
   --out_json out\miura_fit_mp_0_200.json

