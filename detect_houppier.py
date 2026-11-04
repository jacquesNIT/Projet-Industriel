#!/usr/bin/env python3
"""
Détection des arbres individuels sur un MNH LiDAR HD (produit par extract_LidarHD.py) :
sommets de houppier (maximum de hauteur de chaque arbre) puis délimitation des houppiers.

Méthode :
  1. Lissage gaussien du MNH (supprime les maxima parasites dus aux branches).
  2. Sommets = maxima locaux par fenêtre circulaire de taille variable : le diamètre de la
     fenêtre suit la largeur de houppier attendue pour la hauteur du pixel, d'après la
     relation de Popescu & Wynne (2004) pour les pins :
         largeur (m) = 3.75105 - 0.17919 h + 0.01241 h²
     Seuls les pixels plus hauts que --hmin sont candidats ; un plateau de maxima égaux
     compte pour un seul sommet.
  3. Houppiers = ligne de partage des eaux (watershed) sur le MNH lissé inversé, avec un
     marqueur par sommet ; un pixel n'est gardé que s'il dépasse --seuil-houppier x la
     hauteur de son arbre (sépare houppier et sous-bois / sol).

Sorties dans visual_<dossier de données>/ (comme visual_LidarHD.py) :
  <raster>[_zoom_...]_houppiers[_ortho][_hist].png : MNH avec contours des houppiers et sommets
  <raster>[_zoom_...]_houppiers[_ortho][_hist].csv : un arbre par ligne (x, y Lambert-93,
                                                    hauteur, surface et diamètre du houppier)

Exemples :
  # Zoom 200 m x 200 m au centre de la zone, avec l'orthophoto pour contrôle
  #   -> visual_lidar_pissos/mnh_zoom_397900_6355900_398100_6356100_houppiers_ortho.png (+ .csv)
  python detect_houppier.py lidar_pissos/mnh --zoom 397900 6355900 398100 6356100 \\
      --ortho landes_pissos/mosaique_rvb_0.2m.tif

  # Toute la zone (2 km x 2 km) avec histogramme des hauteurs d'arbres
  #   -> visual_lidar_pissos/mosaique_mnh_0.5m_houppiers_hist.png (+ .csv)
  python detect_houppier.py lidar_pissos/mosaique_mnh_0.5m.tif --hist

  # Grille WMS produite par extract_LidarHD.py (même zoom de 200 m)
  python detect_houppier.py lidar_pissos/mnh_wms_0.5m.tif --ortho landes_pissos/mosaique_rvb_0.2m.tif

  # Peuplement plus dense ou jeune : fenêtres plus petites, seuil de hauteur plus bas
  python detect_houppier.py lidar_pissos/mnh --zoom 397900 6355900 398100 6356100 --facteur 0.8 --hmin 1.5
"""

import argparse
import csv
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

from ign_commun import lire, en_rgb, nom_crs, enregistrer_figure, deviner_produit

FIGURE_MAX_PX = 4000   # côté maximal d'un panneau de la figure (px)
MARGE = 10.0           # marge (m) lue autour du zoom pour ne pas couper les arbres du bord


def largeur_houppier(h):
    """Largeur de houppier attendue (m) pour un pin de hauteur h (m), Popescu & Wynne (2004)."""
    return 3.75105 - 0.17919 * h + 0.01241 * h ** 2


def disque(r):
    """Fenêtre circulaire de rayon r pixels."""
    n = int(r)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return xx ** 2 + yy ** 2 <= r ** 2


def detecter_sommets(z, chm, res, hmin, facteur):
    """Maxima locaux à fenêtre variable sur le MNH lissé z.
    Renvoie les positions (lignes, colonnes) des sommets, choisies sur le MNH brut chm."""
    # Pré-sélection rapide : un sommet est forcément un maximum de son voisinage 3x3
    lignes, cols = np.nonzero((z >= hmin) & (z == ndimage.maximum_filter(z, size=3)))
    # Rayon de fenêtre (px) selon la hauteur, arrondi au demi-pixel pour limiter les passes
    rayon = np.maximum(facteur * largeur_houppier(z[lignes, cols]) / 2 / res, 1.5)
    rayon = np.round(rayon * 2) / 2
    marge = int(rayon.max()) + 1 if len(rayon) else 0
    zp = np.pad(z, marge, constant_values=-np.inf)
    garde = np.zeros(len(lignes), bool)
    for r in np.unique(rayon):
        i = np.flatnonzero(rayon == r)
        zi = z[lignes[i], cols[i]]
        ok = np.ones(len(i), bool)
        n = int(r)
        # Le candidat doit dominer (ou égaler) chaque pixel de sa fenêtre circulaire
        for dy, dx in zip(*np.nonzero(disque(r))):
            ok &= zp[lignes[i] + marge + dy - n, cols[i] + marge + dx - n] <= zi
        garde[i] = ok
    sommet = np.zeros(z.shape, bool)
    sommet[lignes[garde], cols[garde]] = True
    # Un plateau de maxima égaux = un seul arbre : on garde son pixel le plus haut
    lab, n = ndimage.label(sommet, structure=np.ones((3, 3)))
    pos = ndimage.maximum_position(chm, lab, np.arange(1, n + 1))
    return np.array(pos, dtype=int).reshape(-1, 2)


def delimiter_houppiers(z, sommets, hauteurs, hmin, seuil):
    """Watershed sur le MNH lissé inversé, un marqueur par sommet.
    Renvoie une image d'étiquettes (0 = hors houppier, i = arbre i)."""
    marqueurs = np.zeros(z.shape, np.int32)
    marqueurs[z < hmin] = -1                       # fond (sol, sous-bois)
    marqueurs[sommets[:, 0], sommets[:, 1]] = np.arange(1, len(sommets) + 1)
    relief = np.clip((z.max() - z) * 100, 0, 65535).astype(np.uint16)   # au cm près
    lab = ndimage.watershed_ift(relief, marqueurs, structure=np.ones((3, 3)))
    lab[lab < 0] = 0
    h_arbre = np.concatenate([[0], hauteurs])[lab]
    lab[(z < hmin) | (z < seuil * h_arbre)] = 0
    return lab


def bords(lab):
    """Pixels de contour des houppiers (un seul pixel de large entre deux houppiers voisins)."""
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return b


def main():
    p = argparse.ArgumentParser(description="Détection des sommets et houppiers sur un MNH LiDAR HD")
    p.add_argument("raster", help="MNH LiDAR HD : GeoTIFF ou dossier de dalles")
    p.add_argument("--zoom", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="emprise à traiter en Lambert-93 (défaut : tout le raster)")
    p.add_argument("--ortho", help="orthophoto RVB (GeoTIFF ou dossier de tuiles) à afficher à côté")
    p.add_argument("--hist", action="store_true", help="afficher l'histogramme des hauteurs d'arbres")
    p.add_argument("--hmin", type=float, default=2.0, help="hauteur minimale d'un arbre en m (défaut 2)")
    p.add_argument("--lissage", type=float, default=0.5,
                   help="écart-type du lissage gaussien en m, 0 pour aucun (défaut 0.5)")
    p.add_argument("--facteur", type=float, default=1.0,
                   help="multiplie la taille des fenêtres de recherche (défaut 1)")
    p.add_argument("--seuil-houppier", type=float, default=0.5,
                   help="fraction de la hauteur de l'arbre sous laquelle on sort du houppier (défaut 0.5)")
    p.add_argument("-o", "--out", help="dossier de sortie (défaut visual_<dossier de données>)")
    p.add_argument("--show", action="store_true", help="afficher aussi la figure")
    a = p.parse_args()

    if deviner_produit(a.raster) != "mnh":
        sys.exit("La détection des houppiers se fait sur un MNH (hauteur au-dessus du sol), "
                 "pas sur un MNT ou un MNS.")

    # Lecture pleine résolution, avec une marge autour du zoom
    lecture = None
    if a.zoom:
        lecture = (a.zoom[0] - MARGE, a.zoom[1] - MARGE, a.zoom[2] + MARGE, a.zoom[3] + MARGE)
    chm, ext, res, crs = lire(a.raster, lecture, max_px=math.inf)
    if chm.count() == 0:
        sys.exit("Aucune donnée valide dans cette zone (nodata partout).")
    chm = chm.filled(0).astype("float32")
    z = ndimage.gaussian_filter(chm, a.lissage / res) if a.lissage > 0 else chm

    # Détection
    sommets = detecter_sommets(z, chm, res, a.hmin, a.facteur)
    if len(sommets) == 0:
        sys.exit(f"Aucun arbre de plus de {a.hmin} m détecté.")
    hauteurs = chm[sommets[:, 0], sommets[:, 1]]
    lab = delimiter_houppiers(z, sommets, hauteurs, a.hmin, a.seuil_houppier)
    surfaces = np.bincount(lab.ravel(), minlength=len(sommets) + 1)[1:] * res ** 2
    xs = ext[0] + (sommets[:, 1] + 0.5) * res
    ys = ext[3] - (sommets[:, 0] + 0.5) * res

    # On ne garde que les arbres dont le sommet est dans la zone demandée (hors marge)
    vue = a.zoom if a.zoom else (ext[0], ext[2], ext[1], ext[3])
    dedans = (xs >= vue[0]) & (xs < vue[2]) & (ys >= vue[1]) & (ys < vue[3])
    xs, ys, hauteurs, surfaces = xs[dedans], ys[dedans], hauteurs[dedans], surfaces[dedans]
    ids = np.flatnonzero(dedans) + 1
    vue = (max(vue[0], ext[0]), max(vue[1], ext[2]), min(vue[2], ext[1]), min(vue[3], ext[3]))
    ha = (vue[2] - vue[0]) * (vue[3] - vue[1]) / 1e4
    diametres = 2 * np.sqrt(surfaces / np.pi)

    print(f"{a.raster} (MNH, {nom_crs(crs)})")
    print(f"Emprise : X {vue[0]:.0f}–{vue[2]:.0f}, Y {vue[1]:.0f}–{vue[3]:.0f} ({ha:.2f} ha), {res:.2f} m/px")
    print(f"{len(xs)} arbres détectés, soit {len(xs) / ha:.0f} arbres/ha")
    if len(xs):
        print(f"Hauteurs : min {hauteurs.min():.1f} m, médiane {np.median(hauteurs):.1f} m, "
              f"max {hauteurs.max():.1f} m")
        print(f"Houppiers : diamètre équivalent médian {np.median(diametres):.1f} m, "
              f"surface médiane {np.median(surfaces):.1f} m²")

    # Figure : MNH (+ ortho) avec contours des houppiers et sommets
    n_cartes = 1 + bool(a.ortho)
    npx = max(vue[2] - vue[0], vue[3] - vue[1]) / res
    taille = min(max(npx, 1050), FIGURE_MAX_PX) / 150          # pouces, à 150 dpi
    s = max((3 * 72 * taille / npx) ** 2, 1)                   # sommet ≈ 3 pixels du MNH
    ratios = [taille] * n_cartes + [taille / 2] * a.hist
    fig, axes = plt.subplots(1, len(ratios), figsize=(sum(ratios) + 2, taille), squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    axes = axes[0]
    contour = np.ma.masked_where(~bords(lab), np.ones(lab.shape))
    vmax = float(np.percentile(chm[chm >= a.hmin], 99.5)) if (chm >= a.hmin).any() else 1

    ax = axes[0]
    im = ax.imshow(chm, cmap="viridis", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
    ax.imshow(contour, cmap=ListedColormap(["white"]), extent=ext, interpolation="nearest", alpha=0.7)
    ax.scatter(xs, ys, s=s, c="red", linewidths=0)
    fig.colorbar(im, ax=ax, label="Hauteur au-dessus du sol (m)", shrink=0.8)
    ax.set_title(f"MNH LiDAR HD : {len(xs)} houppiers détectés")
    ax.set_ylabel("Y Lambert-93 (m)")

    if a.ortho:
        rgb, ext_o, _, _ = lire(a.ortho, vue, bands=[1, 2, 3], max_px=FIGURE_MAX_PX)
        ax2 = axes[1]
        ax2.imshow(en_rgb(rgb), extent=ext_o)
        ax2.imshow(contour, cmap=ListedColormap(["yellow"]), extent=ext, interpolation="nearest", alpha=0.6)
        ax2.scatter(xs, ys, s=s, c="red", linewidths=0)
        ax2.sharex(ax); ax2.sharey(ax)   # zoom synchronisé entre les deux vues
        ax2.set_title("BD ORTHO")

    for axe in axes[:n_cartes]:
        axe.set_xlim(vue[0], vue[2]); axe.set_ylim(vue[1], vue[3])
        axe.set_xlabel("X Lambert-93 (m)")
        axe.ticklabel_format(useOffset=False, style="plain")

    if a.hist and len(xs):
        ax3 = axes[-1]
        ax3.hist(hauteurs, bins=50, color="tab:green")
        ax3.set_box_aspect(1)
        ax3.set_xlabel("Hauteur de l'arbre (m)")
        ax3.set_ylabel("Nombre d'arbres")
        ax3.set_title("Distribution des hauteurs")

    plt.tight_layout()
    options = ["houppiers"] + [o for o in ("ortho", "hist") if getattr(a, o)]
    png = enregistrer_figure(fig, a.raster, a.zoom, options, a.out)

    # Table des arbres
    chemin_csv = png.with_suffix(".csv")
    with open(chemin_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x_l93", "y_l93", "hauteur_m", "surface_houppier_m2", "diametre_houppier_m"])
        for row in zip(ids, xs, ys, hauteurs, surfaces, diametres):
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}",
                        f"{row[4]:.2f}", f"{row[5]:.2f}"])
    print(f"Table des arbres : {chemin_csv}")

    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
