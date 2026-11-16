#Lire avant le tutorial book de Jean-Romain https://r-lidar.github.io/arbor_book/

install.packages('arbor', repos = c('https://r-lidar.r-universe.dev', 'https://cloud.r-project.org'))
install.packages("lidR")
install.packages("ggplot2")
install.packages("RMCC")
install.packages("future")

library(lidR)
library(arbor)
library(ggplot2)
params <- arbor_parameters_default
filter <- "-keep_random_fraction 0.25"

#Chargement nuage, filtration
BSA = readLAS("BSA_2026_06_18.laz")
BSA <- hybrid_homogeneization(BSA)
params$global$cut_above_ground = 0.25
print(BSA)

#Classification points sol et création DTM
BSA <- segment_ground(BSA)
dtm <- lidR::rasterize_terrain(BSA, 0.1, lidR::tin())
plot(dtm)
plot_dtm3d(dtm, bg = "white")

#Segmentation semantique
BSA <- wood_likelihood(BSA, params)
BSA <- segment_semantic(BSA, params)


#Segmentation individuelle et sup. si Ht<15m et buffer 15m
seeds <- find_seeds(BSA, params)
BSA   <- segment_instance(BSA, seeds, params)
BSA <- resolve_oversegmentation(BSA)
BSA <- flag_small_trees(BSA, max_height = 15)
BSA <- flag_buffer(BSA, seeds, -15)
BSA <- lidR::filter_poi(BSA, UserData == ARBORTREE)

#Construction arbres multiples et filtre par qualité reconstru.
qsf <- qsf(BSA, params = params)
log <- qsf_log(qsf)
log

saplings  <- qsf_filter_sapling(qsf)
nomeasure <- qsf_filter_nomeasure(qsf)
goodtrees <- qsf_filter_ok(qsf)

x <- plot_instance(BSA)
plot(goodtrees, add = x, pal = "chocolate4")
plot(nomeasure, add = x, pal = "chocolate4")

plot(qsf)
plot(qsf, pal = "chocolate4") |> lidR::add_dtm3d(dtm)
tm <- qsf_treemap(qsf)

#Filtration et Graph pour bois marchant uniquement
qsf_merch <- qsf_merchantable(qsf)
plot(qsf_merch)
qsf_stem <- qsm_stem(qsf)
plot(qsf_stem)
qsf_merch <- qsm_merchantable(qsf)

las <- qsf_segment_semantic(BSA, qsf)
plot(las)



  