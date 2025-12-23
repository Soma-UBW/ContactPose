import argparse, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---- globals (each worker process) ----
G_JOINTS = None
G_FITTER = None
G_SIDE = None
G_N_POSE_PARAMS = 15
G_SHAPE_SIGMA = 10.0
G_ONLY_FINAL = True

def _to_jsonable(x):
    import numpy as _np
    try:
        import chumpy as _ch
        if isinstance(x, _ch.Ch):
            return _np.array(x.r).tolist()
    except Exception:
        pass
    if x is None:
        return None
    if isinstance(x, (float, int, str, bool)):
        return x
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return str(x)

def _pick_final(out):
    # fit_joints returns like [init_dict, final_dict]
    if isinstance(out, list) and len(out) >= 2 and isinstance(out[-1], dict):
        return out[-1]
    if isinstance(out, dict):
        return out
    return None

def _init_worker(joints_path: str, side: str, n_pose_params: int, shape_sigma: float, only_final: bool):
    global G_JOINTS, G_FITTER, G_SIDE, G_N_POSE_PARAMS, G_SHAPE_SIGMA, G_ONLY_FINAL
    G_JOINTS = np.load(joints_path, mmap_mode="r")  # shared via OS page cache
    G_SIDE = side
    G_N_POSE_PARAMS = n_pose_params
    G_SHAPE_SIGMA = shape_sigma
    G_ONLY_FINAL = only_final

    # import inside worker
    from utilities import mano_fitting as mf
    G_FITTER = mf.MANOFitter()

def _fit_one(frame_idx: int):
    global G_JOINTS, G_FITTER, G_SIDE, G_N_POSE_PARAMS, G_SHAPE_SIGMA, G_ONLY_FINAL
    J = np.asarray(G_JOINTS[frame_idx], dtype=np.float64)  # (21,3)
    both = (None, J) if G_SIDE == "right" else (J, None)
    try:
        out = G_FITTER.fit_joints(
            both,
            n_pose_params=G_N_POSE_PARAMS,
            shape_sigma=G_SHAPE_SIGMA,
            save_filename=None
        )
        if G_ONLY_FINAL:
            final = _pick_final(out)
            return {"frame": frame_idx, "ok": True, "out": _to_jsonable(final)}
        else:
            return {"frame": frame_idx, "ok": True, "out": _to_jsonable(out)}
    except Exception as e:
        return {"frame": frame_idx, "ok": False, "error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints_npy", required=True, help="(T,21,3) .npy")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--only_final", action="store_true", help="最終結果だけ保存（JSONを軽くする）")
    args = ap.parse_args()

    joints = np.load(args.joints_npy, mmap_mode="r")
    assert joints.ndim == 3 and joints.shape[1:] == (21, 3), joints.shape
    T = joints.shape[0]

    start = max(0, args.start)
    end = T if args.end == -1 else min(T, args.end)
    idxs = list(range(start, end, args.stride))

    # keep defaults (do not change param count)
    n_pose_params = 15
    shape_sigma = 10.0

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.joints_npy, args.side, n_pose_params, shape_sigma, bool(args.only_final)),
    ) as ex:
        futures = [ex.submit(_fit_one, i) for i in idxs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fitting(mp)"):
            results.append(fut.result())

    # sort by frame index
    results.sort(key=lambda r: r["frame"])

    payload = {
        "side": args.side,
        "shape": [int(T), 21, 3],
        "range": {"start": start, "end": end, "stride": args.stride},
        "parallel": {"workers": args.workers},
        "params": {"n_pose_params": n_pose_params, "shape_sigma": shape_sigma},
        "only_final": bool(args.only_final),
        "results": results,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"saved: {args.out_json}  ok={ok_n}/{len(results)}")

if __name__ == "__main__":
    main()
