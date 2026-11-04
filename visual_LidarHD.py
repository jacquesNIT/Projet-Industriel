#!/usr/bin/env python3
"""
Visualisation des rasters LiDAR HD (MNH, MNT, MNS) en GeoTIFF Lambert-93
(produits par extract_LidarHD.py).

Le raster peut être une mosaïque ou un dossier de dalles (assemblées à la volée).
La figure est enregistrée en PNG dans visual_<dossier de données>/ à la racine du projet
(ex. lidar_pissos/mnh -> visual_lidar_pissos/mnh.png).

Exemples :
  # Vue d'ensemble du MNH, directement depuis les dalles -> visual_lidar_pissos/mnh.png
  python visual_LidarHD.py lidar_pissos/mnh

  # Zoom sur 200 m x 200 m (coordonnées Lambert-93), au centre de la zone
  #   -> visual_lidar_pissos/mnh_zoom_397900_6355900_398100_6356100.png
  python visual_LidarHD.py lidar_pissos/mnh --zoom 397900 6355900 398100 6356100

  # MNH à côté de l'orthophoto, sur la même emprise, avec histogramme des hauteurs
  python visual_LidarHD.py lidar_pissos/mnh --ortho landes_pissos/mosaique_rvb_0.2m.tif \\
      --zoom 397000 6355000 399000 6357000 --hist

  # MNT avec ombrage du relief -> visual_lidar_pissos/mnt_ombrage.png
  python visual_LidarHD.py lidar_pissos/mnt --ombrage

  # Après extract_LidarHD.py ... --mosaic : mosaïque de la zone landes_pissos
  python visual_LidarHD.py lidar_pissos/mosaique_mnh_0.5m.tif --ortho landes_pissos/mosaique_rvb_0.2m.tif

  # Afficher aussi la figure (zoom synchronisé MNH / ortho), ou choisir un autre dossier
  python visual_LidarHD.py lidar_pissos/mnh --ortho landes_pissos/mosaique_rvb_0.2m.tif --show
  python visual_LidarHD.py lidar_pissos/mnh -o figures
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from ign_commun import lire, en_rgb, nom_crs, enregistrer_figure, deviner_produit


def ombrage(z, res, azimut=315, hauteur=45):
    """Ombrage (hillshade) classique à partir des pentes."""
    z = z.filled(np.nan) if np.ma.isMaskedArray(z) else z
    dy, dx = np.gradient(z, res)
    pente = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(azimut), np.radians(hauteur)
    hs = np.sin(alt) * np.cos(pente) + np.cos(alt) * np.sin(pente) * np.cos(az - aspect)
    return np.ma.masked_invalid(np.clip(hs, 0, 1))


def main():
    p = argparse.ArgumentParser(description="Visualisation MNH / MNT / MNS LiDAR HD")
    p.add_argument("raster", help="GeoTIFF LiDAR (mnh, mnt ou mns) ou dossier de dalles")
    p.add_argument("--zoom", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="emprise à afficher en Lambert-93 (pleine résolution si petite)")
    p.add_argument("--ortho", help="orthophoto RVB (GeoTIFF ou dossier de tuiles) à afficher à côté")
    p.add_argument("--ombrage", action="store_true", help="superposer un ombrage du relief")
    p.add_argument("--vmin", type=float, help="borne basse de l'échelle de couleurs")
    p.add_argument("--vmax", type=float, help="borne haute de l'échelle de couleurs")
    p.add_argument("--hist", action="store_true", help="afficher l'histogramme des hauteurs")
    p.add_argument("-o", "--out", help="dossier de sortie (défaut visual_<dossier de données>)")
    p.add_argument("--show", action="store_true", help="afficher aussi la figure")
    a = p.parse_args()

    produit = deviner_produit(a.raster)
    z, ext, res, crs = lire(a.raster, a.zoom)

    if z.count() == 0:
        sys.exit("Aucune donnée valide dans cette zone (nodata partout).")

    # Statistiques
    print(f"{a.raster} ({produit.upper()}, {nom_crs(crs)})")
    print(f"Emprise : X {ext[0]:.0f}–{ext[1]:.0f}, Y {ext[2]:.0f}–{ext[3]:.0f}")
    print(f"Affiché : {z.shape[1]}x{z.shape[0]} px à {res:.2f} m/px")
    print(f"Hauteurs : min {z.min():.2f} m, médiane {np.ma.median(z):.2f} m, max {z.max():.2f} m, "
          f"{100 * z.count() / z.size:.0f} % de pixels valides")

    # Échelle de couleurs adaptée au produit
    if produit == "mnh":
        cmap, label = "viridis", "Hauteur au-dessus du sol (m)"
        vmin = a.vmin if a.vmin is not None else 0
        vmax = a.vmax if a.vmax is not None else float(np.percentile(z.compressed(), 99.5))
    else:
        cmap, label = "terrain", "Altitude (m)"
        vmin = a.vmin if a.vmin is not None else float(np.percentile(z.compressed(), 0.5))
        vmax = a.vmax if a.vmax is not None else float(np.percentile(z.compressed(), 99.5))

    n = 1 + bool(a.ortho) + bool(a.hist)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    axes = axes[0]
    k = 0

    # Raster LiDAR
    ax = axes[k]; k += 1
    im = ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
    if a.ombrage:
        ax.imshow(ombrage(z, res), cmap="gray", alpha=0.35, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_title(f"{produit.upper()} LiDAR HD")
    ax.set_xlabel("X Lambert-93 (m)")
    ax.set_ylabel("Y Lambert-93 (m)")

    # Orthophoto sur la même emprise
    if a.ortho:
        bbox = (ext[0], ext[2], ext[1], ext[3])
        rgb, ext_o, _, _ = lire(a.ortho, bbox, bands=[1, 2, 3])
        ax2 = axes[k]; k += 1
        ax2.imshow(en_rgb(rgb), extent=ext_o)
        ax2.set_xlim(ext[0], ext[1]); ax2.set_ylim(ext[2], ext[3])
        ax2.sharex(ax); ax2.sharey(ax)   # zoom synchronisé entre les deux vues
        ax2.set_title("BD ORTHO")
        ax2.set_xlabel("X Lambert-93 (m)")

    # Histogramme
    if a.hist:
        ax3 = axes[k]
        ax3.hist(z.compressed(), bins=100, range=(vmin, vmax), color="tab:green")
        ax3.set_xlabel(label)
        ax3.set_ylabel("Nombre de pixels")
        ax3.set_title("Distribution des valeurs")

    plt.tight_layout()
    options = [o for o in ("ortho", "ombrage", "hist") if getattr(a, o)]
    enregistrer_figure(fig, a.raster, a.zoom, options, a.out)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()