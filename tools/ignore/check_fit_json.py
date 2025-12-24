import argparse, json, sys, math
from pathlib import Path
import numpy as np

# repo importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilities import misc as mutils
from utilities import mano_fitting as mf

def quat_norm(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4: 
        return np.nan
    return float(np.linalg.norm(q))

def apply_T(T4, X):
    # X: (N,3)
    Xh = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    Y = (T4 @ Xh.T).T[:, :3]
    return Y

def mTc_to_mat(mTc):
    # mTc = {"translation":[3], "rotation":[4]} ; rotation is assumed wxyz (as in your json)
    t = np.array(mTc["translation"], dtype=np.float64).reshape(3)
    q = np.array(mTc["rotation"], dtype=np.float64).reshape(4)
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        R = np.eye(3)
    else:
        w, x, y, z = w/n, x/n, y/n, z/n
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def load_final(json_path, frame=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # single-frame: {"out":[{init},{final}]}
    if isinstance(data, dict) and "out" in data and isinstance(data["out"], list):
        last = data["out"][-1]
        if not isinstance(last, dict):
            raise ValueError("single-frame json: out[-1] is not dict")
        return last

    # all-frames: {"results":[{"frame":i,"ok":...,"out":...}, ...]}
    if isinstance(data, dict) and "results" in data:
        if frame is None:
            raise ValueError("all-frames json requires --frame or --frames")
        # find
        for r in data["results"]:
            if int(r.get("frame",-999999)) == int(frame):
                if not r.get("ok", True):
                    raise ValueError(f"frame={frame} ok=false: {r.get('error')}")
                out = r.get("out")
                if isinstance(out, dict):
                    return out
                if isinstance(out, list) and len(out) and isinstance(out[-1], dict):
                    return out[-1]
                raise ValueError("results[].out is neither dict nor list-of-dict")
        raise ValueError(f"frame={frame} not found in results")

    raise ValueError("Unknown json format (needs 'out' or 'results')")

def build_mano_and_joints(side, pose, betas, n_pose_params=15):
    # initialize MANOFitter (sets internal dicts)
    _ = mf.MANOFitter()
    hand_idx = 1 if side == "right" else 0
    mano_dicts = getattr(mf.MANOFitter, "_mano_dicts", None)
    if mano_dicts is None:
        raise RuntimeError("MANOFitter._mano_dicts is None")

    m = mutils.load_mano_model(mano_dicts[hand_idx], ncomps=n_pose_params, flat_hand_mean=False)

    # safe set
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    bb = np.zeros(m.betas.size, dtype=np.float64)
    bb[:min(bb.size, betas.size)] = betas[:min(bb.size, betas.size)]
    m.betas[:] = bb

    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    pp = np.zeros(m.pose.size, dtype=np.float64)
    pp[:min(pp.size, pose.size)] = pose[:min(pp.size, pose.size)]
    m.pose[:] = pp

    mano_joints = mutils.mano_joints_with_fingertips(m)  # list of chumpy vec3
    Jm = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in mano_joints], dtype=np.float64)
    return m, Jm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--joints_npy", required=True)
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--n_pose_params", type=int, default=15)
    ap.add_argument("--frames", default="0", help="e.g. '0,10,20' or '0:200:10'")
    ap.add_argument("--max_print", type=int, default=20)
    args = ap.parse_args()

    A = np.load(args.joints_npy)
    assert A.ndim == 3 and A.shape[1:] == (21,3), A.shape
    T = A.shape[0]

    # parse frames
    frames = []
    s = args.frames.strip()
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        if len(parts) == 2:
            a, b = map(int, parts); step = 1
        else:
            a, b, step = map(int, parts[:3])
        frames = list(range(max(0,a), min(T,b), max(1,step)))
    else:
        frames = [int(x) for x in s.split(",") if x.strip() != ""]
    frames = [f for f in frames if 0 <= f < T]
    if not frames:
        raise ValueError("No valid frames parsed")

    stats = []
    printed = 0

    for f in frames:
        final = load_final(args.json, frame=f)

        # structural checks
        pose = final.get("pose")
        betas = final.get("betas")
        mTc = final.get("mTc")
        valid = final.get("valid", True)

        ok_struct = (pose is not None) and (betas is not None) and (mTc is not None)
        if not ok_struct:
            stats.append({"frame": f, "ok": False, "reason": "missing keys"})
            continue

        pose = np.asarray(pose, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        q = mTc.get("rotation", None)
        t = mTc.get("translation", None)

        ok_num = np.isfinite(pose).all() and np.isfinite(betas).all()
        ok_num = ok_num and (q is not None) and (t is not None)
        if ok_num:
            q = np.asarray(q, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64)
            ok_num = ok_num and np.isfinite(q).all() and np.isfinite(t).all()

        qn = quat_norm(q) if q is not None else np.nan

        # compute RMSE in the same frame as optimization
        try:
            # mimic fit_joints: convert input joints as if "openpose order" (same as what fit used)
            cp = mutils.openpose2mano(A[f].astype(np.float64))
            Tm = mTc_to_mat(mTc)
            cp_mano = apply_T(Tm, cp)

            _, manoJ = build_mano_and_joints(args.side, pose, betas, n_pose_params=args.n_pose_params)

            rmse = float(np.sqrt(np.mean(np.sum((manoJ - cp_mano)**2, axis=1))))
            ok_rmse = np.isfinite(rmse)
        except Exception as e:
            rmse = np.nan
            ok_rmse = False
            err = str(e)

        ok = bool(ok_struct and ok_num and ok_rmse)

        rec = {
            "frame": f,
            "ok": ok,
            "valid_flag": bool(valid),
            "pose_len": int(pose.size),
            "betas_len": int(betas.size),
            "quat_norm": qn,
            "rmse_mano_frame": rmse,
        }
        if not ok and not ok_rmse:
            rec["error"] = err
        stats.append(rec)

        if printed < args.max_print:
            print(rec)
            printed += 1

    # summary
    oks = [r for r in stats if r.get("ok")]
    rmse_vals = [r["rmse_mano_frame"] for r in oks if np.isfinite(r.get("rmse_mano_frame", np.nan))]
    print("\n--- summary ---")
    print("frames checked:", len(stats), "ok:", len(oks))
    if rmse_vals:
        print("rmse median:", float(np.median(rmse_vals)), "mean:", float(np.mean(rmse_vals)))
        print("quat_norm median:", float(np.median([r["quat_norm"] for r in oks if np.isfinite(r["quat_norm"])])))
    else:
        print("rmse: no valid values")

if __name__ == "__main__":
    main()
