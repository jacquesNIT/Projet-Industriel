#!/usr/bin/env python3
"""
Visualisation des orthophotos BD ORTHO en GeoTIFF Lambert-93 (produites par extract_BDortho.py).

Le raster peut être une mosaïque ou un dossier de tuiles (assemblées à la volée).
La figure est enregistrée en PNG dans visual_<dossier de données>/ à la racine du projet
(ex. landes_pissos/mosaique_rvb_0.2m.tif -> visual_landes_pissos/mosaique_rvb_0.2m.png).

Exemples :
  # Vue d'ensemble de la zone de Pissos -> visual_landes_pissos/mosaique_rvb_0.2m.png
  python visual_BDortho.py landes_pissos/mosaique_rvb_0.2m.tif

  # Directement depuis les tuiles -> visual_landes_pissos/tuiles.png
  python visual_BDortho.py landes_pissos/tuiles

  # Zoom sur 200 m x 200 m (coordonnées Lambert-93), au centre de la zone
  #   -> visual_landes_pissos/mosaique_rvb_0.2m_zoom_397900_6355900_398100_6356100.png
  python visual_BDortho.py landes_pissos/mosaique_rvb_0.2m.tif --zoom 397900 6355900 398100 6356100

  # Afficher aussi la figure, ou choisir un autre dossier de sortie
  python visual_BDortho.py landes_pissos/mosaique_rvb_0.2m.tif --show
  python visual_BDortho.py landes_pissos/mosaique_rvb_0.2m.tif -o figures
"""

import argparse

import matplotlib.pyplot as plt

from ign_commun import lire, en_rgb, nom_crs, enregistrer_figure


def main():
    p = argparse.ArgumentParser(description="Visualisation BD ORTHO")
    p.add_argument("raster", help="GeoTIFF RVB ou dossier de tuiles")
    p.add_argument("--zoom", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="emprise à afficher en Lambert-93 (pleine résolution si petite)")
    p.add_argument("-o", "--out", help="dossier de sortie (défaut visual_<dossier de données>)")
    p.add_argument("--show", action="store_true", help="afficher aussi la figure")
    a = p.parse_args()

    img, ext, res, crs = lire(a.raster, a.zoom, bands=[1, 2, 3])

    print(f"{a.raster} ({nom_crs(crs)})")
    print(f"Emprise : X {ext[0]:.0f}–{ext[1]:.0f}, Y {ext[2]:.0f}–{ext[3]:.0f}")
    print(f"Affiché : {img.shape[2]}x{img.shape[1]} px à {res:.2f} m/px")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(en_rgb(img), extent=ext)
    ax.set_title("BD ORTHO")
    ax.set_xlabel("X Lambert-93 (m)")
    ax.set_ylabel("Y Lambert-93 (m)")

    plt.tight_layout()
    enregistrer_figure(fig, a.raster, a.zoom, out=a.out)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
