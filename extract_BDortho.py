#!/usr/bin/env python3
"""
Extraction de la BD ORTHO (IGN) via le WMS-Raster de la Géoplateforme.

- Découpe automatiquement la zone en tuiles (le serveur limite à 5010 px par côté)
- Enregistre chaque tuile en GeoTIFF géoréférencé (Lambert-93, EPSG:2154)
- Reprend là où il s'est arrêté (tuiles déjà présentes ignorées)
- Assemble optionnellement les tuiles en une seule mosaïque

Sortie : <dossier>/tuiles/<couche>_<x>_<y>_<res>m.tif  (+ <dossier>/mosaique_<couche>_<res>m.tif)

Exemples :
  # Landes (Pissos), 2 km x 2 km en Lambert-93 (xmin ymin xmax ymax, en mètres), 20 cm/pixel,
  # même emprise que lidar_pissos (c'est la commande qui a produit landes_pissos)
  python extract_BDortho.py --bbox 397000 6355000 399000 6357000 -o landes_pissos --mosaic

  # Bordeaux, 2 km x 2 km
  python extract_BDortho.py --bbox 416000 6420000 418000 6422000 -o ortho_bdx --mosaic

  # Emprise en longitude/latitude (WGS84), 50 cm/pixel
  python extract_BDortho.py --lonlat -0.60 44.79 -0.56 44.82 --res 0.5 -o ortho_bdx_50cm --mosaic

  # Infrarouge couleur sur la zone de Pissos
  python extract_BDortho.py --bbox 397000 6355000 399000 6357000 --layer irc -o landes_pissos_irc --mosaic
"""

import argparse
import math
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from ign_commun import (CRS, WMS_URL, WMS_MAX_PX, http_get, ajouter_emprise, emprise_l93,
                        caler_emprise, mosaique)

LAYERS = {
    "rvb": "HR.ORTHOIMAGERY.ORTHOPHOTOS",   # BD ORTHO couleur, 20 cm
    "rvb50": "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO",  # BD ORTHO couleur, 50 cm
    "irc": "ORTHOIMAGERY.ORTHOPHOTOS.IRC",  # infrarouge couleur
}


def tuile_wms(layer, bounds, width, height):
    """Télécharge une tuile WMS et renvoie un tableau numpy (bandes, lignes, colonnes)."""
    xmin, ymin, xmax, ymax = bounds
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": CRS,
        # En WMS 1.3.0 avec EPSG:2154, l'ordre des axes est bien x,y (est, nord)
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/geotiff",
    }
    r = http_get(WMS_URL, params=params)
    ctype = r.headers.get("Content-Type", "")
    # Le serveur renvoie un XML d'erreur en cas de paramètre invalide
    if "image" not in ctype:
        raise RuntimeError(f"Erreur WMS ({ctype}) : {r.text[:300]}")
    with MemoryFile(r.content) as mem, mem.open() as src:
        return src.read()


def ecrire_geotiff(path, data, bounds):
    """Écrit le tableau en GeoTIFF avec le géoréférencement calculé à partir de la bbox."""
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "count": count,
        "height": height,
        "width": width,
        "crs": CRS,
        "transform": from_bounds(*bounds, width=width, height=height),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def telecharger(bbox, res, layer_key, out_dir):
    layer = LAYERS[layer_key]
    tiles_dir = Path(out_dir) / "tuiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = caler_emprise(bbox, res)
    tile_size = WMS_MAX_PX * res  # taille d'une tuile en mètres
    nx = math.ceil((xmax - xmin) / tile_size)
    ny = math.ceil((ymax - ymin) / tile_size)
    total_px = round((xmax - xmin) / res) * round((ymax - ymin) / res)
    print(f"Couche : {layer}")
    print(f"Emprise L93 : {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}  |  résolution {res} m")
    print(f"{nx} x {ny} tuile(s), ~{total_px / 1e6:.0f} Mpx au total\n")

    paths = []
    for j in range(ny):
        for i in range(nx):
            tx0 = xmin + i * tile_size
            ty0 = ymin + j * tile_size
            tx1 = min(tx0 + tile_size, xmax)
            ty1 = min(ty0 + tile_size, ymax)
            w = round((tx1 - tx0) / res)
            h = round((ty1 - ty0) / res)
            path = tiles_dir / f"{layer_key}_{int(tx0)}_{int(ty0)}_{res}m.tif"
            paths.append(path)
            n = j * nx + i + 1
            if path.exists():
                print(f"[{n}/{nx * ny}] {path.name} déjà présent, ignoré")
                continue
            print(f"[{n}/{nx * ny}] {path.name} ({w}x{h} px)")
            data = tuile_wms(layer, (tx0, ty0, tx1, ty1), w, h)
            ecrire_geotiff(path, data, (tx0, ty0, tx1, ty1))
    return paths


def main():
    p = argparse.ArgumentParser(description="Extraction BD ORTHO IGN (WMS)")
    ajouter_emprise(p)
    p.add_argument("--res", type=float, default=0.2, help="résolution en m/pixel (défaut 0.2)")
    p.add_argument("--layer", choices=LAYERS, default="rvb", help="couche (défaut rvb)")
    p.add_argument("-o", "--out", default="bdortho", help="dossier de sortie")
    p.add_argument("--mosaic", action="store_true", help="assembler les tuiles en un seul GeoTIFF")
    a = p.parse_args()

    paths = telecharger(emprise_l93(a), a.res, a.layer, a.out)
    if a.mosaic and paths:
        mosaique(paths, Path(a.out) / f"mosaique_{a.layer}_{a.res}m.tif")


if __name__ == "__main__":
    main()
