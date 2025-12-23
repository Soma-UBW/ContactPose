import argparse, json
import numpy as np

def to_jsonable(x):
    import numpy as _np
    try:
        import chumpy as _ch
        if isinstance(x, _ch.Ch):
            return _np.array(x.r).tolist()
    except Exception:
        pass
    if x is None:
        return None
    if isinstance(x, (float,int,str,bool)):
        return x
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, (list,tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k,v in x.items()}
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints_npy", required=True, help="(T,21,3)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joints = np.load(args.joints_npy)
    assert joints.ndim == 3 and joints.shape[1:] == (21,3), joints.shape
    J = joints[args.frame].astype(np.float64)

    from utilities import mano_fitting as mf

    if not hasattr(mf, "MANOFitter"):
        raise RuntimeError("utilities.mano_fitting に MANOFitter が見つかりません")

    # 初期化（引数がある実装もあるので、まずは引数なしで試す）
    try:
        fitter = mf.MANOFitter()
    except TypeError:
        # もし引数必須なら、signatureを出して止める
        import inspect
        raise RuntimeError(f"MANOFitter の初期化引数を確認してください: {inspect.signature(mf.MANOFitter)}")

    # 右手/左手を (left, right) タプルで渡す
    both = (None, J) if args.side == "right" else (J, None)

    # フィット実行：メソッド名が実装差で違うことがあるので候補順に試す
    out = None
    last_err = None
    for mname in ["fit", "fit_joints", "fit_to_joints", "run", "optimize"]:
        if hasattr(fitter, mname) and callable(getattr(fitter, mname)):
            try:
                out = getattr(fitter, mname)(both)
                used = mname
                break
            except Exception as e:
                last_err = e

    if out is None:
        raise RuntimeError(f"MANOFitter の実行メソッドが見つからない/失敗。last_err={last_err}")

    result = {
        "class": "MANOFitter",
        "method": used,
        "frame": args.frame,
        "side": args.side,
        "out": to_jsonable(out),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", args.out_json)

if __name__ == "__main__":
    main()
