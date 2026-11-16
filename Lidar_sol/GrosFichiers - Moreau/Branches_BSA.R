# Extraction des insertions de branches sur la partie MARCHANDE du tronc,
# pour chaque arbre d'un QSF arbor
#
# À lancer APRÈS Arbor_BSA.R, dans la même session R (il faut les objets `goodtrees` et `dtm`) :
#   source("Arbor_BSA.R", echo = TRUE)
#   source("Branches_BSA.R", echo = TRUE)
#
# Rappels sur la structure d'un QSM arbor (une ligne = un cylindre) :
#   - startX/Y/Z, endX/Y/Z : extrémités du cylindre (repère du nuage, ici LOCAL au scanner)
#   - cyl_ID / parent_ID   : graphe (le début d'un cylindre = la fin de son parent)
#   - axis_ID              : 1 = tronc, puis un identifiant par branche
#   - branch_order         : 1 = tronc, 2 = branche portée par le tronc, 3 = ramification...
#   - radius (m), quality  : >= 3 = ajustement géométrique réel, 2 = estimé (pipe model),
#                            1 = prolongement jusqu'au sol, 0 = inconnu

# ---------------------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------------------
# Zone marchande du tronc (à valider avec la scierie / M. Moreau)
H_SOUCHE        <- 0.15  # m : hauteur de la souche (coupe)
DIAM_DECOUPE_CM <- 20    # cm : diamètre fin bout de la découpe sciage (7 pour inclure le bois d'industrie)
LISSAGE_N       <- 5     # nb de cylindres du tronc pour lisser le profil de diamètre (médiane glissante)

LONGUEUR_DIRECTION <- 0.5   # m : longueur de branche utilisée pour estimer sa direction
FICHIER_BRANCHES   <- "BSA_branches_marchand.csv"
FICHIER_ARBRES     <- "BSA_zone_marchande.csv"

# Critères d'une branche "fiable" (squelette assez long pour ne pas être du bruit)
N_CYL_MIN     <- 3     # nombre minimal de cylindres
LONG_MIN      <- 1     # m : longueur minimale (branche + ramifications)
INCL_MIN      <- -20   # degrés : on écarte les "branches" qui plongent fortement vers le sol
# Au-delà de ce rapport diamètre branche / diamètre tronc : branche maîtresse (grosse branche
# charpentière ou fourche). Attention, ce rapport repose sur des diamètres ESTIMÉS (pipe model).
RATIO_MAITRESSE <- 0.5

# ---------------------------------------------------------------------------------------
# Fonction : zone marchande du tronc d'un QSM
#   Va de H_SOUCHE jusqu'au premier point où le diamètre (lissé) du tronc passe sous
#   DIAM_DECOUPE_CM. Renvoie un data.frame d'une ligne.
# ---------------------------------------------------------------------------------------
zone_marchande <- function(q, ground_z = NULL) {
  tree_id <- attr(q, "id")
  if (is.null(tree_id)) tree_id <- NA
  d <- as.data.frame(q)

  stem <- d[d$axis_ID == 1, ]
  stem <- stem[order(stem$dist_to_root), ]
  if (is.null(ground_z) || is.na(ground_z)) ground_z <- stem$startZ[1]

  h_bas <- stem$startZ - ground_z            # hauteur du bas de chaque cylindre
  diam  <- 2 * stem$radius * 100             # cm

  # Lissage du profil (les cylindres individuels sont bruités)
  k <- min(LISSAGE_N, nrow(stem))
  if (k %% 2 == 0) k <- k - 1
  diam_l <- if (k >= 3) as.vector(stats::runmed(diam, k, endrule = "median")) else diam

  au_dessus <- which(h_bas >= H_SOUCHE)
  if (length(au_dessus) == 0) au_dessus <- seq_len(nrow(stem))
  i0 <- au_dessus[1]                                   # premier cylindre au-dessus de la souche
  sous <- au_dessus[diam_l[au_dessus] < DIAM_DECOUPE_CM]

  if (length(sous) == 0) {
    # Le tronc ne passe jamais sous la découpe : tout le tronc reconstruit est marchand
    h_decoupe <- max(stem$endZ) - ground_z
    i1 <- nrow(stem)
    atteinte <- FALSE
  } else {
    i1 <- sous[1]
    h_decoupe <- h_bas[i1]
    atteinte <- TRUE
  }
  marchand <- diam_l[i0] >= DIAM_DECOUPE_CM

  data.frame(
    treeID               = tree_id,
    hauteur_arbre_m      = max(d$endZ) - ground_z,
    diam_souche_cm       = diam_l[i0],                      # diamètre (lissé) au-dessus de la souche
    marchand             = marchand,                        # FALSE = trop fin dès la souche
    h_souche_m           = H_SOUCHE,
    h_decoupe_m          = if (marchand) h_decoupe else H_SOUCHE,
    longueur_marchande_m = if (marchand) max(0, h_decoupe - H_SOUCHE) else 0,
    diam_decoupe_cm      = diam_l[i1],
    decoupe_atteinte     = atteinte                         # FALSE = sommet du tronc encore > découpe
  )
}

# ---------------------------------------------------------------------------------------
# Fonction : branches insérées sur le tronc d'un QSM (toutes hauteurs)
#   q        : un objet qsm (ex. goodtrees[[1]])
#   ground_z : altitude du sol au pied de l'arbre (NULL = pied du tronc selon arbor)
#   dir_len  : longueur (m) le long de la branche pour calculer sa direction
# Renvoie un data.frame, une ligne par branche d'ordre 2
# ---------------------------------------------------------------------------------------
branches_qsm <- function(q, ground_z = NULL, dir_len = LONGUEUR_DIRECTION) {
  tree_id <- attr(q, "id")
  if (is.null(tree_id)) tree_id <- NA
  d <- as.data.frame(q)

  stem <- d[d$axis_ID == 1, ]
  if (nrow(stem) == 0) return(NULL)
  if (is.null(ground_z) || is.na(ground_z)) ground_z <- stem$startZ[1]

  d$len <- sqrt((d$endX - d$startX)^2 + (d$endY - d$startY)^2 + (d$endZ - d$startZ)^2)

  # Ligne du parent de chaque cylindre
  ip <- match(d$parent_ID, d$cyl_ID)
  parent_axis <- d$axis_ID[ip]

  # Premier cylindre de chaque branche portée par le tronc :
  # cylindre hors tronc dont le parent est sur le tronc
  first <- which(d$axis_ID != 1 & !is.na(ip) & parent_axis == 1)
  if (length(first) == 0) return(NULL)

  res <- lapply(first, function(i) {
    a <- d$axis_ID[i]
    b <- d[d$axis_ID == a, ]
    b <- b[order(b$dist_to_root), ]
    s <- d[ip[i], ]                       # cylindre du tronc qui porte la branche

    # Point d'insertion = début du premier cylindre de la branche (sur l'axe du tronc)
    p0 <- c(d$startX[i], d$startY[i], d$startZ[i])

    # Direction : de l'insertion jusqu'au point situé à ~dir_len le long de la branche
    k  <- which(cumsum(b$len) >= dir_len)[1]
    if (is.na(k)) k <- nrow(b)
    p1 <- c(b$endX[k], b$endY[k], b$endZ[k])
    v  <- p1 - p0
    vh <- sqrt(v[1]^2 + v[2]^2)

    # Axe local du tronc
    u <- c(s$endX - s$startX, s$endY - s$startY, s$endZ - s$startZ)

    cosang <- sum(u * v) / (sqrt(sum(u^2)) * sqrt(sum(v^2)))
    cosang <- max(-1, min(1, cosang))

    data.frame(
      treeID            = tree_id,
      branche_axis_ID   = a,
      x_insertion       = p0[1],
      y_insertion       = p0[2],
      z_insertion       = p0[3],
      h_insertion_m     = p0[3] - ground_z,          # hauteur au-dessus du sol
      dist_to_root_m    = d$dist_to_root[i],         # distance le long du tronc
      diam_base_cm      = 2 * d$radius[i] * 100,     # diamètre du 1er cylindre de la branche
      diam_tronc_cm     = 2 * s$radius * 100,        # diamètre du tronc à l'insertion
      azimut_deg        = (atan2(v[1], v[2]) * 180 / pi) %% 360,  # 0 = +Y, 90 = +X (repère local !)
      inclinaison_deg   = atan2(v[3], vh) * 180 / pi,             # / horizontale, > 0 = vers le haut
      angle_tronc_deg   = acos(cosang) * 180 / pi,                # angle entre tronc et branche
      longueur_axe_m    = sum(b$len),                # longueur de la branche seule
      longueur_totale_m = d$subtree_length[i] + d$len[i],  # branche + ses ramifications
      n_cylindres       = nrow(b),
      qualite_base      = d$quality[i],
      part_mesuree      = mean(b$quality >= 3),      # fraction de cylindres réellement ajustés
      diam_mesure       = d$quality[i] >= 3          # FALSE = diamètre estimé (pipe model)
    )
  })
  do.call(rbind, res)
}

# ---------------------------------------------------------------------------------------
# Application à tous les arbres bien reconstruits
# ---------------------------------------------------------------------------------------
if (!exists("goodtrees")) stop("Objet `goodtrees` introuvable : lancer d'abord Arbor_BSA.R")

# Altitude du sol au pied de chaque arbre, lue dans le MNT si disponible
sol_au_pied <- function(q) {
  d <- as.data.frame(q)
  pied <- d[d$axis_ID == 1, ][1, ]
  if (exists("dtm")) {
    z <- terra::extract(dtm, cbind(pied$startX, pied$startY))[[1]]
    if (!is.na(z)) return(z)
  }
  pied$startZ
}

res <- lapply(goodtrees, function(q) {
  tryCatch({
    z0 <- sol_au_pied(q)
    list(zone = zone_marchande(q, z0), branches = branches_qsm(q, z0))
  }, error = function(e) {
    message("Arbre ", attr(q, "id"), " ignoré : ", conditionMessage(e))
    NULL
  })
})
zones           <- do.call(rbind, lapply(res, `[[`, "zone"))
branches_toutes <- do.call(rbind, lapply(res, `[[`, "branches"))
rownames(zones) <- rownames(branches_toutes) <- NULL

# La position et la direction viennent du squelette (donc des points) ; le diamètre, lui,
# est presque toujours estimé. On qualifie donc les branches sur leur géométrie.
branches_toutes$ratio_diam <- branches_toutes$diam_base_cm / branches_toutes$diam_tronc_cm
branches_toutes$maitresse  <- branches_toutes$ratio_diam >= RATIO_MAITRESSE
branches_toutes$fiable     <- branches_toutes$n_cylindres >= N_CYL_MIN &
                              branches_toutes$longueur_totale_m >= LONG_MIN &
                              branches_toutes$inclinaison_deg > INCL_MIN

# On ne garde que les branches insérées dans la zone marchande de leur arbre
iz <- match(branches_toutes$treeID, zones$treeID)
dans_zone <- zones$marchand[iz] &
             branches_toutes$h_insertion_m >= zones$h_souche_m[iz] &
             branches_toutes$h_insertion_m <= zones$h_decoupe_m[iz]
branches <- branches_toutes[dans_zone, ]
rownames(branches) <- NULL
# Position de la branche dans la zone marchande (0 = souche, 1 = découpe)
branches$position_relative <- (branches$h_insertion_m - zones$h_souche_m[match(branches$treeID, zones$treeID)]) /
                              zones$longueur_marchande_m[match(branches$treeID, zones$treeID)]

cat("\nZone marchande : souche à", H_SOUCHE, "m, découpe à", DIAM_DECOUPE_CM, "cm fin bout\n")
cat(sum(zones$marchand), "arbres marchands sur", nrow(zones), "\n")
cat(nrow(branches), "branches dans la zone marchande (sur", nrow(branches_toutes), "au total)",
    "dont", sum(branches$fiable), "fiables et", sum(branches$fiable & branches$maitresse), "maîtresses\n\n")

# Résumé par arbre (tous les arbres, même sans branche dans la zone)
f <- factor(branches$treeID, levels = zones$treeID)
premiere <- function(h) if (all(is.na(h))) NA else min(h, na.rm = TRUE)
par_arbre <- function(x, fun, def) as.vector(tapply(x, f, fun, default = def))
h_branche   <- ifelse(branches$fiable & !branches$maitresse, branches$h_insertion_m, NA)
h_maitresse <- ifelse(branches$fiable &  branches$maitresse, branches$h_insertion_m, NA)
d_fiable    <- ifelse(branches$fiable & !branches$maitresse, branches$diam_base_cm, NA)

resume <- cbind(zones,
  n_branches                  = par_arbre(branches$fiable, length, 0),
  n_fiables                   = par_arbre(branches$fiable, sum, 0),
  n_maitresses                = par_arbre(branches$fiable & branches$maitresse, sum, 0),
  h_premiere_branche_fiable_m = par_arbre(h_branche, premiere, NA),   # = longueur de bille sans branche
  h_premiere_maitresse_m      = par_arbre(h_maitresse, premiere, NA),
  diam_max_branche_cm         = par_arbre(d_fiable, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE), NA)
)
print(resume, digits = 3)

write.csv(branches, FICHIER_BRANCHES, row.names = FALSE)
write.csv(resume,   FICHIER_ARBRES,   row.names = FALSE)
cat("\nBranches écrites dans", FICHIER_BRANCHES, "\nRésumé par arbre écrit dans", FICHIER_ARBRES, "\n")

# Pour ne garder que les branches fiables hors maîtresses :
# branches_ok <- subset(branches, fiable & !maitresse)
# Toutes les branches de l'arbre (y compris hors zone marchande) sont dans `branches_toutes`.
