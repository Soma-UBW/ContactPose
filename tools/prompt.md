# csv to npy and json

cd C:\Users\shimi\Documents\aolab\ContactPose

.\.venv310\Scripts\Activate.ps1

cd .\ContactPose

$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

$env:PYTHONPATH = (Get-Location).Path + ";" + $env:PYTHONPATH

python tools\fit_all_frames_mp.py `
  --joints_npy out\miura_joints21.npy `
  --side right `
  --start 0 --end 200 `
  --workers 8 `
  --only_final `
  --out_json out\miura_fit_mp_0_200.json

python tools\fit_all_frames_mp.py `
  --joints_npy out\miura_joints21.npy `
  --side right `
  --workers 8 `
  --only_final `
  --out_json out\miura_fit_mp_all.json

# json visualize

python tools\vis_mano.py `
  --json out\miura_fit_mp_all.json `
  --joints_npy out\miura_joints21.npy `
  --frame 123 `
  --side right

python tools\vis_mano.py `
  --json out\miura_mano_fit_frame0.json `
  --joints_npy out\miura_joints21.npy `
  --frame 0 `
  --side right `
  --show_mano_frame

# npy visualize
python tools\vis_joints_npy.py `
--npy out\miura_joints21.npy `
--frame 0