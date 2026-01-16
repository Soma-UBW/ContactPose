# setup

cd C:\Users\shimi\Documents\aolab\ContactPose

.\.venv310\Scripts\Activate.ps1

cd .\ContactPose

$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

$env:PYTHONPATH = (Get-Location).Path + ";" + $env:PYTHONPATH

# csv to npy

python tools\csv_to_npy.py `
  --csv C:\Users\shimi\Documents\aolab\Contactpose\csv\05_akimoto.csv `
  --out_npy out\akimoto.npy `
  --unit cm `
  --make_right

# npy visualize
python tools\vis_joints_npy.py `
--npy out\miura.npy `
--frame 3200

# npy to json

python tools\fit_one_frame.py `
  --joints_npy out\watanabe.npy `
  --frame 0 `
  --side right `
  --out_json out\akimoto_frame0.json

python tools\fit_all_frames_mp.py `
  --joints_npy out\watanabe.npy `
  --side right `
  --start 0 --end 10000 `
  --workers 8 `
  --only_final `
  --out_json out\watanabe_frame0-10k.json

python tools\fit_all_frames_mp.py `
  --joints_npy out\miura.npy `
  --side right `
  --workers 8 `
  --only_final `
  --out_json out\miura_all.json

# json visualize

python tools\vis_mano.py `
  --json out\miura_frame0-10k.json `
  --joints_npy out\miura.npy `
  --frame 3130 `
  --side right `
  --show_mano_frame
