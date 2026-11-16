"""Build the cross-modal manifests (protocol: docs/58 section 13).

One manifest per modality with a fixed, seeded, class-stratified sample and a
matched ``real'`` partner used by the P2 null. Everything here is deterministic:
the same raw archives and the same seed reproduce the same 600 rows.

    python scripts/120_prepare_crossmodal.py --modality all

Sources (see protocol 13.2):
  mri   Figshare 10.6084/m9.figshare.1512427  (.mat, cjdata.PID available)
  oct   Mendeley 10.17632/rscbjbr9sj.3        (ZhangLabData.zip, official test split)
  derm  Harvard Dataverse 10.7910/DVN/DBW86T  (lesion_id available)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST_DIR = DATA / "manifests"

SEED = 20261114
N_TARGET = 600

MRI_LABEL = {1: "meningioma", 2: "glioma", 3: "pituitary_tumor"}

OCT_LABEL = {
    "CNV": "choroidal_neovascularization",
    "DME": "diabetic_macular_edema",
    "DRUSEN": "drusen",
    "NORMAL": "normal",
}

DERM_LABEL = {
    "akiec": "actinic_keratosis",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic_nevus",
    "vasc": "vascular_lesion",
}


# --------------------------------------------------------------------------- #
# mri
# --------------------------------------------------------------------------- #
def _load_cjdata(path: Path) -> dict:
    """Read ``cjdata`` from either a MATLAB v5/v6 file or the v7.3 HDF5 release.

    The Figshare release is v7.3, which ``scipy.io.loadmat`` refuses. h5py stores
    MATLAB arrays transposed, so ``image`` is flipped back to MATLAB orientation
    to match the conversion recipe in the dataset README.
    """
    try:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path)["cjdata"].items()
                if not k.startswith("__")}
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            g = f["cjdata"]
            return {
                "PID": np.asarray(g["PID"]),
                "label": np.asarray(g["label"]),
                "image": np.asarray(g["image"]).T,
            }


def _decode_pid(value) -> str:
    """PID is a 6-character code stored as char codes, not an integer."""
    arr = np.asarray(value).ravel()
    if arr.size and arr.dtype.kind in "ui" and arr.size <= 32:
        text = "".join(chr(int(c)) for c in arr).strip()
        if text.isdigit() or text.isalnum():
            return text
    if arr.dtype.kind in "US":
        return "".join(str(x) for x in arr)
    return str(int(arr.squeeze()))


def prepare_mri(n: int = N_TARGET) -> pd.DataFrame:
    raw = DATA / "brain_mri_raw"
    out = DATA / "brain_mri" / "images"
    out.mkdir(parents=True, exist_ok=True)

    zips = sorted(raw.glob("brainTumorDataPublic_*.zip"))
    if not zips:
        raise FileNotFoundError(f"no Figshare archives under {raw}")
    extracted = raw / "mat"
    extracted.mkdir(exist_ok=True)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".mat"):
                    continue
                target = extracted / Path(name).name
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())

    rows = []
    for mat in sorted(extracted.glob("*.mat")):
        cj = _load_cjdata(mat)
        pid = _decode_pid(cj["PID"])
        label = MRI_LABEL[int(np.asarray(cj["label"]).squeeze())]
        img = np.asarray(cj["image"], dtype=np.float64)
        lo, hi = float(img.min()), float(img.max())
        if hi <= lo:
            continue
        img8 = (255.0 / (hi - lo) * (img - lo)).astype(np.uint8)
        stem = mat.stem
        png = out / f"{stem}.png"
        if not png.exists():
            Image.fromarray(img8).save(png)
        rows.append({
            "image_id": f"mri_{stem}",
            "modality": "mri",
            "label": label,
            "split_unit": f"pid_{pid}",
            "image_relpath": str(png.relative_to(DATA)).replace("\\", "/"),
        })

    df = pd.DataFrame(rows)
    print(f"[mri] {len(df)} slices, {df.split_unit.nunique()} patients, "
          f"classes={df.label.value_counts().to_dict()}")
    # Round-robin over patients; this arm trains nothing, so the only role of
    # unit-awareness is to minimise patient clustering in the 600.
    return _sample(df, n)


# --------------------------------------------------------------------------- #
# oct
# --------------------------------------------------------------------------- #
def prepare_oct(n: int = N_TARGET) -> pd.DataFrame:
    raw = DATA / "oct_raw"
    src = raw / "ZhangLabData.zip"
    if not src.exists():
        raise FileNotFoundError(src)
    out = DATA / "oct" / "images"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with zipfile.ZipFile(src) as zf:
        names = [x for x in zf.namelist()
                 if "/test/" in x.lower().replace("\\", "/")
                 and x.lower().endswith((".jpeg", ".jpg", ".png"))
                 and not x.endswith("/")]
        for name in names:
            parts = Path(name).parts
            cls = next((p for p in parts if p.upper() in OCT_LABEL), None)
            if cls is None:
                continue
            label = OCT_LABEL[cls.upper()]
            target = out / Path(name).name
            if not target.exists():
                with zf.open(name) as s, open(target, "wb") as d:
                    d.write(s.read())
            rows.append({
                "image_id": f"oct_{Path(name).stem}",
                "modality": "oct",
                "label": label,
                # Kermany does not distribute patient identifiers; the official
                # test split is documented as patient-disjoint from training but
                # within-test clustering is unverifiable. Deviation logged in
                # docs/58 section 13.10.
                "split_unit": "",
                "image_relpath": str(target.relative_to(DATA)).replace("\\", "/"),
            })

    df = pd.DataFrame(rows)
    print(f"[oct] {len(df)} test images, classes={df.label.value_counts().to_dict()}")
    return _sample(df, n)


# --------------------------------------------------------------------------- #
# derm
# --------------------------------------------------------------------------- #
def prepare_derm(n: int = N_TARGET) -> pd.DataFrame:
    raw = DATA / "ham10000_raw"
    out = DATA / "ham10000" / "images"
    out.mkdir(parents=True, exist_ok=True)

    for z in sorted(raw.glob("HAM10000_images_part_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".jpg"):
                    continue
                target = out / Path(name).name
                if not target.exists():
                    with zf.open(name) as s, open(target, "wb") as d:
                        d.write(s.read())

    meta_path = raw / "HAM10000_metadata.tab"
    meta = pd.read_csv(meta_path, sep="\t")
    meta = meta.drop_duplicates(subset=["image_id"])
    meta["label"] = meta["dx"].map(DERM_LABEL)
    meta = meta[meta["label"].notna()]
    # A handful of lesions carry two diagnoses in HAM10000 and would otherwise be
    # sampled once in each of two class subsets, putting the same lesion in the
    # panel twice. Keep one row per lesion, lowest image_id first.
    meta = meta.sort_values("image_id").drop_duplicates(subset=["lesion_id"], keep="first")

    rows = []
    for rec in meta.itertuples(index=False):
        png = out / f"{rec.image_id}.jpg"
        if not png.exists():
            continue
        rows.append({
            "image_id": f"derm_{rec.image_id}",
            "modality": "derm",
            "label": rec.label,
            "split_unit": str(rec.lesion_id),
            "image_relpath": str(png.relative_to(DATA)).replace("\\", "/"),
        })

    df = pd.DataFrame(rows)
    print(f"[derm] {len(df)} images, {df.split_unit.nunique()} lesions, "
          f"classes={df.label.value_counts().to_dict()}")
    # One image per lesion follows from the round-robin, because nv has 6,705
    # images over far more lesions than the 90-image quota: duplicates of the
    # same lesion never enter the sample.
    return _sample(df, n)


# --------------------------------------------------------------------------- #
# shared sampling + pairing
# --------------------------------------------------------------------------- #
def _round_robin(sub: pd.DataFrame, quota: int) -> pd.DataFrame:
    """Take one image per split unit per pass, until the quota is met.

    For a fixed n this maximises the number of distinct patients/lesions, which
    minimises the design effect the cluster bootstrap has to pay for. A flat
    per-unit cap cannot do that: 233 MRI patients cannot supply 600 slices at two
    each, and a cap high enough to reach the quota lets one patient contribute
    six slices while another contributes none.
    """
    by_unit = {}
    for unit, grp in sub.groupby("split_unit", dropna=False):
        by_unit[str(unit)] = grp.sort_values("image_id").reset_index(drop=True)
    order = sorted(by_unit, key=lambda u: hashlib.sha1(u.encode()).hexdigest())

    take, depth = [], 0
    while len(take) < quota:
        progressed = False
        for unit in order:
            if len(take) >= quota:
                break
            if depth < len(by_unit[unit]):
                take.append(by_unit[unit].iloc[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return pd.DataFrame(take).reset_index(drop=True)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Equal-per-class allocation, deterministic, class presence preserved."""
    # Equal allocation; the remainder goes to the largest classes first, so the
    # modality's dominant class absorbs it rather than whichever name sorts first.
    counts = df["label"].value_counts()
    classes = list(counts.index)
    quota = {c: n // len(classes) for c in classes}
    for c in classes[: n - sum(quota.values())]:
        quota[c] += 1

    picked = []
    for c in classes:
        take = _round_robin(df[df["label"] == c], quota[c])
        if len(take) < quota[c]:
            print(f"[warn] {c}: only {len(take)} of {quota[c]} requested")
        picked.append(take.drop(columns=[k for k in ("_h",) if k in take.columns]))
    out = pd.concat(picked, ignore_index=True).sort_values("image_id")

    # real' partner: same label, different split unit (when units exist)
    out = out.reset_index(drop=True)
    partners = []
    for i, row in out.iterrows():
        same = out[(out["label"] == row["label"]) & (out.index != i)]
        if row["split_unit"]:
            same = same[same["split_unit"] != row["split_unit"]]
        if same.empty:
            partners.append("")
            continue
        partners.append(str(same.iloc[(i + 1) % len(same)]["image_id"]))
    out["pair_id"] = partners
    if (out["pair_id"] == "").any():
        raise RuntimeError("empty real' partner: a class has a single split unit")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="all",
                    choices=["all", "mri", "oct", "derm"])
    ap.add_argument("--n", type=int, default=N_TARGET)
    args = ap.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    builders = {"mri": prepare_mri, "oct": prepare_oct, "derm": prepare_derm}
    todo = list(builders) if args.modality == "all" else [args.modality]

    summary = {}
    for m in todo:
        df = builders[m](args.n)
        path = MANIFEST_DIR / f"crossmodal_{m}.csv"
        df.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        summary[m] = {
            "rows": int(len(df)),
            "classes": {k: int(v) for k, v in df["label"].value_counts().items()},
            "split_units": int(df["split_unit"].replace("", np.nan).nunique()),
            "sha256": digest,
        }
        print(f"[{m}] wrote {path} sha256={digest[:16]}...")

    (MANIFEST_DIR / "crossmodal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
