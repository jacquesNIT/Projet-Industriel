"""
Fonctions communes aux scripts d'extraction et de visualisation des données IGN
(BD ORTHO et LiDAR HD), toutes en Lambert-93 (EPSG:2154).

Utilisé par : extract_BDortho.py, extract_LidarHD.py, visual_BDortho.py, visual_LidarHD.py,
              detect_houppier.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

CRS = "EPSG:2154"                           # Lambert-93
WMS_URL = "https://data.geopf.fr/wms-r/wms"
WMS_MAX_PX = 5000                           # limite serveur WMS : 5010 px par côté
AFFICHAGE_MAX_PX = 2000                     # côté maximal affiché (évite de saturer la mémoire)
PROJET = Path(__file__).resolve().parent   # racine du projet (dossiers de données)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def http_get(url, params=None, timeout=180, retries=4, stream=False):
    """GET avec nouvelles tentatives (réseau, timeout, erreur HTTP)."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"    échec ({e}), nouvel essai dans {wait} s...")
            time.sleep(wait)


def ajouter_emprise(parser):
    """Ajoute les options --bbox (Lambert-93) / --lonlat (WGS84) à un parseur argparse."""
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="emprise en Lambert-93 (mètres)")
    g.add_argument("--lonlat", nargs=4, type=float, metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"),
                   help="emprise en WGS84 (degrés)")


def emprise_l93(args):
    """Renvoie l'emprise (xmin, ymin, xmax, ymax) en Lambert-93 à partir de --bbox ou --lonlat."""
    if args.lonlat:
        bbox = transform_bounds("EPSG:4326", CRS, *args.lonlat, densify_pts=21)
    else:
        bbox = args.bbox
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        sys.exit("Emprise invalide : il faut xmin < xmax et ymin < ymax.")
    return tuple(bbox)


def caler_emprise(bbox, res):
    """Cale l'emprise sur la grille de résolution pour avoir des pixels entiers."""
    xmin, ymin, xmax, ymax = bbox
    return (math.floor(xmin / res) * res, math.floor(ymin / res) * res,
            math.ceil(xmax / res) * res, math.ceil(ymax / res) * res)


def mosaique(paths, out_path):
    """Assemble des GeoTIFF en un seul (les autres fichiers, ex. LAZ, sont ignorés)."""
    from rasterio.merge import merge

    paths = [p for p in paths if Path(p).suffix.lower() in (".tif", ".tiff")]
    if not paths:
        return
    srcs = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(driver="GTiff", height=data.shape[1], width=data.shape[2], crs=CRS,
                       transform=transform, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    finally:
        for s in srcs:
            s.close()
    print(f"\nMosaïque écrite : {out_path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def lister_rasters(path):
    """Un GeoTIFF, ou tous les GeoTIFF d'un dossier (dalles / tuiles)."""
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.glob("*.tif"))
        if not paths:
            sys.exit(f"Aucun GeoTIFF dans {path}.")
        return paths
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")
    return [path]


def lire(path, zoom=None, bands=1, max_px=AFFICHAGE_MAX_PX):
    """Lit un GeoTIFF, ou un dossier de dalles assemblées à la volée, sur toute son emprise
    ou sur la fenêtre zoom (xmin, ymin, xmax, ymax), sous-échantillonné si trop grand.
    Renvoie (tableau masqué, emprise (xmin, xmax, ymin, ymax), résolution affichée, CRS)."""
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in lister_rasters(path)]
    try:
        xmin = min(s.bounds.left for s in srcs)
        ymin = min(s.bounds.bottom for s in srcs)
        xmax = max(s.bounds.right for s in srcs)
        ymax = max(s.bounds.top for s in srcs)
        if zoom:
            # On rogne la fenêtre aux limites des données
            xmin, ymin = max(xmin, zoom[0]), max(ymin, zoom[1])
            xmax, ymax = min(xmax, zoom[2]), min(ymax, zoom[3])
        if xmax <= xmin or ymax <= ymin:
            sys.exit(f"La zone demandée est hors de {path}.")
        res = srcs[0].res[0]
        res *= max(max(xmax - xmin, ymax - ymin) / res / max_px, 1)
        indexes = [bands] if isinstance(bands, int) else list(bands)
        data, transform = merge(srcs, bounds=(xmin, ymin, xmax, ymax), res=res,
                                indexes=indexes, resampling=Resampling.average, masked=True)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    if isinstance(bands, int):
        data = data[0]
    h, w = data.shape[-2:]
    ext = (transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f)
    return data, ext, res, crs


def deviner_produit(path):
    """Produit LiDAR HD (mnh, mnt, mns) d'après le chemin du raster (défaut mnh)."""
    name = Path(path).as_posix().lower()
    for p in ("mnh", "mnt", "mns"):
        if p in name:
            return p
    return "mnh"


def dossier_visual(raster):
    """Dossier visual_<dossier de données> à la racine du projet, d'après le raster d'origine
    (ex. lidar_pissos/mnh -> visual_lidar_pissos, landes_pissos/tuiles -> visual_landes_pissos)."""
    p = Path(raster).resolve()
    try:
        parts = p.relative_to(PROJET).parts
        nom = parts[0] if len(parts) > 1 else p.stem
    except ValueError:   # raster hors du projet : on prend son dossier
        nom = p.name if p.is_dir() else p.parent.name
    return PROJET / f"visual_{nom}"


def enregistrer_figure(fig, raster, zoom=None, options=(), out=None):
    """Enregistre la figure en PNG dans out (défaut : visual_<dossier de données>).
    Nom : <raster>[_zoom_xmin_ymin_xmax_ymax][_<option>...].png"""
    nom = Path(raster).stem
    if zoom:
        nom += "_zoom_" + "_".join(f"{v:.0f}" for v in zoom)
    nom += "".join(f"_{o}" for o in options)
    d = Path(out) if out else dossier_visual(raster)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{nom}.png"
    fig.savefig(path, dpi=150)
    print(f"Figure enregistrée : {path}")
    return path


def nom_crs(crs):
    """Nom court d'un CRS (les dalles LiDAR HD ont un Lambert-93 sans code EPSG reconnu)."""
    if crs is None:
        return "CRS inconnu"
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    nom = crs.wkt.split('"')[1]   # PROJCS["<nom>", ...
    return f"{nom}, sans code EPSG"


def en_rgb(data):
    """Tableau (3, lignes, colonnes) -> image (lignes, colonnes, 3) affichable par imshow."""
    rgb = np.ma.filled(data, 0).transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(rgb.max(), 1), 0, 1)
    return rgb
