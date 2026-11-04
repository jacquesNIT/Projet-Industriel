#!/usr/bin/env python3
"""
Extraction des données LiDAR HD de l'IGN (Géoplateforme) sur une emprise Lambert-93.

Deux modes :

1) dalles : télécharge les dalles officielles 1 km x 1 km (GeoTIFF natifs, sans
   rééchantillonnage) trouvées via la couche WFS de métadonnées LiDAR HD.
     produits : mnh (hauteur de canopée), mnt (sol), mns (surface), npl (nuage de points LAZ)
   Sortie : <dossier>/<produit>/LHD_FXX_*.tif  (+ <dossier>/mosaique_<produit>_0.5m.tif)

   # Landes (Pissos), même emprise que landes_pissos (BD ORTHO)
   python extract_LidarHD.py dalles --bbox 397000 6355000 399000 6357000 --produits mnh mnt \\
       -o lidar_pissos --mosaic

2) wms : récupère directement une grille de hauteurs (float32) sur l'emprise exacte,
   via le WMS-Raster (pratique pour une petite zone, pas besoin de découper/assembler).
     produits : mnh, mnt
   Sortie : <dossier>/<produit>_wms_<res>m.tif

   # Zoom 200 m x 200 m au centre de landes_pissos -> lidar_pissos/mnh_wms_0.5m.tif
   python extract_LidarHD.py wms --bbox 397900 6355900 398100 6356100 --produit mnh -o lidar_pissos

Les deux modes acceptent --lonlat à la place de --bbox et produisent des GeoTIFF en
EPSG:2154, superposables à la BD ORTHO (voir visual_LidarHD.py --ortho).
"""

import argparse
import math
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from ign_commun import (CRS, WMS_URL, WMS_MAX_PX, http_get, ajouter_emprise, emprise_l93,
                        caler_emprise, mosaique)

WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_LAYER = "IGNF_LIDAR-HD_METADONNEE:metadata"   # une entité par dalle de 1 km
WMS_LAYERS = {
    "mnh": "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
    "mnt": "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
}
NODATA = -99999.0


# ---------------------------------------------------------------------------
# Mode 1 : dalles natives via le WFS de métadonnées
# ---------------------------------------------------------------------------
def sommets(coords):
    """Parcourt les sommets (x, y) d'une géométrie GeoJSON, quel que soit son type."""
    if isinstance(coords[0], (int, float)):
        yield coords
    else:
        for c in coords:
            yield from sommets(c)


def recouvre(feat, bbox, tol=1.0):
    """Vrai si la dalle recouvre réellement l'emprise, et ne fait pas que toucher son bord."""
    xs, ys = zip(*((p[0], p[1]) for p in sommets(feat["geometry"]["coordinates"])))
    xmin, ymin, xmax, ymax = bbox
    return (min(max(xs), xmax) - max(min(xs), xmin) > tol
            and min(max(ys), ymax) - max(min(ys), ymin) > tol)


def lister_dalles(bbox):
    """Renvoie la liste des entités (dalles) du WFS qui intersectent l'emprise."""
    xmin, ymin, xmax, ymax = bbox
    feats, start, page = [], 0, 1000
    while True:
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": WFS_LAYER,
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": CRS,
            "BBOX": f"{xmin},{ymin},{xmax},{ymax},{CRS}",
            "COUNT": page,
            "STARTINDEX": start,
        }
        r = http_get(WFS_URL, params=params)
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"Réponse WFS inattendue :\n{r.text[:500]}")
        batch = data.get("features", [])
        feats += batch
        if len(batch) < page:
            # Le filtre BBOX du WFS renvoie aussi les dalles voisines qui touchent le bord
            return [f for f in feats if recouvre(f, bbox)]
        start += page


def url_produit(props, produit):
    """Trouve l'URL de téléchargement du produit dans les attributs de la dalle."""
    url = props.get(f"url_{produit}")
    if url:
        return url
    # Repli si l'IGN renomme l'attribut : on cherche un champ url_* contenant le produit
    for k, v in props.items():
        if k.lower().startswith("url") and produit in k.lower() and v:
            return v
    return None


def nom_fichier(url, produit, i):
    """Nom de fichier : paramètre FILENAME= de l'URL si présent, sinon fin du chemin."""
    q = parse_qs(urlparse(url).query)
    for key in ("FILENAME", "filename"):
        if key in q:
            return Path(q[key][0]).name
    name = Path(urlparse(url).path).name
    if name and "." in name:
        return name
    ext = ".copc.laz" if produit == "npl" else ".tif"
    return f"{produit}_{i:04d}{ext}"


def fixer_crs(path):
    """Les GeoTIFF servis par l'IGN ont un CRS Lambert-93 non reconnu comme EPSG:2154 : on le réécrit."""
    if path.suffix.lower() not in (".tif", ".tiff"):
        return
    with rasterio.open(path, "r+") as dst:
        if dst.crs is None or dst.crs.to_epsg() != 2154:
            dst.crs = CRS


def telecharger_dalles(bbox, produits, out_dir):
    feats = lister_dalles(bbox)
    print(f"{len(feats)} dalle(s) LiDAR HD sur l'emprise\n")
    if not feats:
        print("Aucune dalle : la zone n'est peut-être pas encore couverte.")
        return {}

    # Affiche les attributs de la première dalle pour contrôle
    print("Attributs disponibles :", ", ".join(feats[0]["properties"].keys()), "\n")

    fichiers = {p: [] for p in produits}
    for produit in produits:
        d = Path(out_dir) / produit
        d.mkdir(parents=True, exist_ok=True)
        manquantes = 0
        for i, f in enumerate(feats):
            url = url_produit(f["properties"], produit)
            if not url:
                manquantes += 1
                continue
            path = d / nom_fichier(url, produit, i)
            fichiers[produit].append(path)
            n = f"{i + 1}/{len(feats)}"
            if path.exists():
                print(f"[{produit} {n}] {path.name} déjà présent, ignoré")
            else:
                print(f"[{produit} {n}] {path.name}")
                r = http_get(url, stream=True, timeout=600)
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.rename(path)
            fixer_crs(path)
        if manquantes:
            print(f"[{produit}] {manquantes} dalle(s) sans URL pour ce produit (non encore produite ?)")
    return fichiers


# ---------------------------------------------------------------------------
# Mode 2 : grille float32 sur emprise exacte via WMS
# ---------------------------------------------------------------------------
def tuile_wms(layer, bounds, width, height):
    """Télécharge une tuile WMS de valeurs brutes et renvoie un tableau float32 (lignes, colonnes)."""
    xmin, ymin, xmax, ymax = bounds
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": CRS,
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/x-bil;bits=32",   # valeurs brutes en float32
    }
    r = http_get(WMS_URL, params=params)
    if "xml" in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"Erreur WMS : {r.text[:300]}")
    expected = width * height * 4
    if len(r.content) != expected:
        raise RuntimeError(f"Taille inattendue ({len(r.content)} octets au lieu de {expected})")
    arr = np.frombuffer(r.content, dtype="<f4").reshape(height, width).copy()
    arr[arr <= -9999] = NODATA
    return arr


def telecharger_wms(bbox, produit, res, out_dir):
    layer = WMS_LAYERS[produit]
    xmin, ymin, xmax, ymax = caler_emprise(bbox, res)
    W, H = round((xmax - xmin) / res), round((ymax - ymin) / res)
    out = np.full((H, W), NODATA, dtype="float32")

    tile = WMS_MAX_PX * res
    nx, ny = math.ceil((xmax - xmin) / tile), math.ceil((ymax - ymin) / tile)
    print(f"Couche : {layer}")
    print(f"Emprise L93 : {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}  |  résolution {res} m")
    print(f"{W}x{H} px, {nx * ny} requête(s)\n")
    for j in range(ny):
        for i in range(nx):
            tx0, ty0 = xmin + i * tile, ymin + j * tile
            tx1, ty1 = min(tx0 + tile, xmax), min(ty0 + tile, ymax)
            w, h = round((tx1 - tx0) / res), round((ty1 - ty0) / res)
            print(f"[{j * nx + i + 1}/{nx * ny}] tuile {w}x{h} px")
            a = tuile_wms(layer, (tx0, ty0, tx1, ty1), w, h)
            c0 = i * WMS_MAX_PX
            r0 = H - (j * WMS_MAX_PX + h)   # les lignes partent du haut (nord)
            out[r0:r0 + h, c0:c0 + w] = a

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{produit}_wms_{res}m.tif"
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=1,
                       width=W, height=H, crs=CRS, nodata=NODATA,
                       transform=from_bounds(xmin, ymin, xmax, ymax, W, H),
                       compress="deflate", tiled=True) as dst:
        dst.write(out, 1)
    valid = out[out != NODATA]
    if valid.size:
        print(f"\n{path} : min {valid.min():.2f} m, max {valid.max():.2f} m, "
              f"{100 * valid.size / out.size:.0f} % de pixels valides")
    else:
        print(f"\n{path} : aucune donnée (zone non couverte ?)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Extraction LiDAR HD IGN (WFS + WMS)")
    sub = p.add_subparsers(dest="mode", required=True)

    pd = sub.add_parser("dalles", help="dalles natives 1 km (WFS + téléchargement)")
    ajouter_emprise(pd)
    pd.add_argument("--produits", nargs="+", default=["mnh"],
                    choices=["mnh", "mnt", "mns", "npl"], help="produits (défaut mnh)")
    pd.add_argument("-o", "--out", default="lidarhd", help="dossier de sortie")
    pd.add_argument("--mosaic", action="store_true", help="assembler les dalles raster de chaque produit")

    pw = sub.add_parser("wms", help="grille float32 sur emprise exacte (WMS)")
    ajouter_emprise(pw)
    pw.add_argument("--produit", default="mnh", choices=list(WMS_LAYERS), help="produit (défaut mnh)")
    pw.add_argument("--res", type=float, default=0.5, help="résolution en m/pixel (défaut et natif 0.5)")
    pw.add_argument("-o", "--out", default="lidarhd", help="dossier de sortie")

    a = p.parse_args()
    bbox = emprise_l93(a)

    if a.mode == "dalles":
        fichiers = telecharger_dalles(bbox, a.produits, a.out)
        if a.mosaic:
            for prod, paths in fichiers.items():
                tifs = [q for q in paths if q.suffix.lower() == ".tif"]
                if tifs:
                    with rasterio.open(tifs[0]) as src:
                        res = src.res[0]
                    mosaique(tifs, Path(a.out) / f"mosaique_{prod}_{res}m.tif")
    else:
        telecharger_wms(bbox, a.produit, a.res, a.out)


if __name__ == "__main__":
    main()
