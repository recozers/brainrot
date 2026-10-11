# /// script
# requires-python = ">=3.10"
# dependencies = ["nilearn", "numpy", "nibabel"]
# ///
"""Export fsaverage5 brain mesh + 6-region vertex map.

Writes:
  ios/BrainRot/Resources/brain_mesh.json   (for the iOS app)
  backend/region_map.json                  (per-vertex region id, used by app.py)

Run:  uv run backend/export_mesh.py
Vertex order = left hemisphere (10242) then right (10242), same as TRIBE v2 output.
"""
import json
from pathlib import Path

import numpy as np
from nilearn import datasets, surface

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ["nature_places", "faces_social", "movement", "reading", "sound_speech", "language"]

# (destrieux label, hemisphere 'L'/'R'/'B') -> region. Order matters: later rules don't override earlier.
RULES = {
    "nature_places": [  # PPA / RSC / OPA
        ("G_oc-temp_med-Parahip", "B"), ("S_collat_transv_ant", "B"), ("S_collat_transv_post", "B"),
        ("S_oc-temp_med_and_Lingual", "B"), ("G_cingul-Post-ventral", "B"), ("S_parieto_occipital", "B"),
        ("S_oc_sup_and_transversal", "B"),
    ],
    "reading": [  # VWFA: left occipitotemporal sulcus + left fusiform
        ("S_oc-temp_lat", "L"), ("G_oc-temp_lat-fusifor", "L"),
    ],
    "faces_social": [  # FFA (right fusiform), STS, TPJ
        ("G_oc-temp_lat-fusifor", "R"), ("S_temporal_sup", "R"), ("G_pariet_inf-Angular", "R"),
        ("G_pariet_inf-Supramar", "R"), ("S_oc-temp_lat", "R"),
    ],
    "movement": [  # MT+/EBA lateral occipitotemporal
        ("G_occipital_middle", "B"), ("S_occipital_ant", "B"), ("S_oc_middle_and_Lunatus", "B"),
        ("G_temporal_inf", "B"),
    ],
    "sound_speech": [  # Heschl / STG / planum temporale
        ("G_temp_sup-G_T_transv", "B"), ("S_temporal_transverse", "B"), ("G_temp_sup-Plan_tempo", "B"),
        ("G_temp_sup-Plan_polar", "B"), ("G_temp_sup-Lateral", "R"), ("Lat_Fis-post", "B"),
    ],
    "language": [  # left IFG + left temporal language network
        ("G_front_inf-Opercular", "L"), ("G_front_inf-Triangul", "L"), ("G_front_inf-Orbital", "L"),
        ("G_temp_sup-Lateral", "L"), ("G_temporal_middle", "L"), ("S_temporal_sup", "L"),
        ("G_pariet_inf-Angular", "L"), ("S_front_inf", "L"),
    ],
}


def region_map():
    d = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in d["labels"]]
    idx = {n: i for i, n in enumerate(labels)}
    maps = {"L": np.asarray(d["map_left"]), "R": np.asarray(d["map_right"])}
    out = {h: np.full(len(m), -1, dtype=int) for h, m in maps.items()}
    for rid, name in enumerate(REGIONS):
        for lab, hemi in RULES[name]:
            for h in (["L", "R"] if hemi == "B" else [hemi]):
                sel = (maps[h] == idx[lab]) & (out[h] == -1)
                out[h][sel] = rid
    return np.concatenate([out["L"], out["R"]])


def main():
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    verts, faces, offset = [], [], 0
    blend = 0.3
    for hemi, sign in (("left", -1), ("right", 1)):
        pial = surface.load_surf_mesh(fs[f"pial_{hemi}"])
        infl = surface.load_surf_mesh(fs[f"infl_{hemi}"])
        pc, ic = np.asarray(pial.coordinates, float), np.asarray(infl.coordinates, float)
        # align inflated to pial centroid before blending
        ic = ic - ic.mean(0) + pc.mean(0)
        v = (1 - blend) * pc + blend * ic
        v[:, 0] += sign * 4.0  # separate hemispheres a bit
        verts.append(v)
        faces.append(np.asarray(pial.faces, int) + offset)
        offset += len(v)
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    V -= (V.max(0) + V.min(0)) / 2
    V /= np.linalg.norm(V, axis=1).max()
    # vertex normals
    N = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12

    reg = region_map()
    assert len(reg) == len(V) == 20484, (len(reg), len(V))
    counts = {REGIONS[i]: int((reg == i).sum()) for i in range(len(REGIONS))}
    print("vertices", len(V), "faces", len(F), "region counts", counts, "unassigned", int((reg == -1).sum()))

    mesh = {
        "vertices": [round(float(x), 4) for x in V.ravel()],
        "normals": [round(float(x), 4) for x in N.ravel()],
        "faces": [int(x) for x in F.ravel()],
        "region": [int(x) for x in reg],
        "regions": REGIONS,
    }
    out = ROOT / "ios/BrainRot/Resources/brain_mesh.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mesh, separators=(",", ":")))
    print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
    rm = ROOT / "backend/region_map.json"
    rm.write_text(json.dumps({"regions": REGIONS, "region": [int(x) for x in reg]}, separators=(",", ":")))
    print("wrote", rm)


if __name__ == "__main__":
    main()
