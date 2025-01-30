# DBSCAN_scratch_R

# Nous supposons pour le calcul de la complexité :
# Nombre de clusters = log(Nombre de Points)
# Nombre de voisins qu'un point possède = log(n/log(n))

library(ggplot2)
library(MASS)
if (!require("mclust")) install.packages("mclust", dependencies = TRUE)
library(mclust) # Pour le calcul de l'ARI (Adjusted Rand Index)

# Définition de la classe Point
Point <- setRefClass("Point",
                     fields = list(
                       coordinates = "numeric",
                       type = "numeric",  # 1:core point, 2:border point, 3:outlier
                       neighboringPoints = "numeric",  # Vecteur numérique
                       cluster = "numeric"  # 1, 2, 3, 4, ... nom du cluster
                     )
)

# Fonction pour calculer les distances euclidiennes et classifier les points
# Complexité : O(n^2) - Chaque point est comparé à tous les autres pour calculer la distance
EuclDist <- function(x, p, eps, MinPts) {
  NeighboringPoints <- numeric()
  type <- 0
  
  for (j in 1:nrow(x)) {
    dist <- sqrt(sum((x[j, ] - p)^2))
    if (dist < eps) {
      NeighboringPoints <- c(NeighboringPoints, j)
    }
  }
  
  if (length(NeighboringPoints) > MinPts) {
    type <- 1  # Core point
  } else if (length(NeighboringPoints) > 1) {
    type <- 2  # Border point
  } else {
    type <- 3  # Outlier
  }
  
  return(list(NeighboringPoints = NeighboringPoints, type = type))
}

# Fonction pour trouver les points appartenant à un cluster donné
# Complexité : O(n × m) - Avec m étant le nombre moyen de voisins par point
findClusterPoints <- function(x, currentCluster, points, position, eps, MinPts) {
  ClusterMembers <- points[[position]]$neighboringPoints
  i <- 1
  
  while (i <= length(ClusterMembers)) {
    expansionPoint <- ClusterMembers[i] # Point d'expansion
    
    if (points[[expansionPoint]]$cluster == -1) { # Point bordure non assigné
      points[[expansionPoint]]$cluster <- currentCluster
    } else if (points[[expansionPoint]]$cluster == 0) { # Point central non assigné
      points[[expansionPoint]]$cluster <- currentCluster
      ClusterMembers <- c(ClusterMembers, points[[expansionPoint]]$neighboringPoints)
    }
    
    i <- i + 1
  }
}

# Fonction principale DBSCAN
# Complexité totale : O(n^2 + n × m) - Dominée par les calculs de distance et l'expansion des clusters
DBscan <- function(x, eps, MinPts) {
  start_time <- Sys.time() # Début du chronométrage
  
  currentCluster <- 0
  points <- list()
  
  # Étape 1 : Initialisation des points
  # Complexité : O(n^2) - Calcul des distances pour chaque point
  for (i in 1:nrow(x)) {
    p <- x[i, ]
    result <- EuclDist(x, p, eps, MinPts)
    points[[i]] <- Point$new(
      coordinates = p,
      type = result$type,
      neighboringPoints = result$NeighboringPoints,
      cluster = 1 - result$type
    )
  }
  
  # Étape 2 : Assignation des clusters
  # Complexité : O(n × m) - Expansion des clusters à partir des points centraux
  for (i in 1:length(points)) {
    if (points[[i]]$cluster == 0) {
      currentCluster <- currentCluster + 1
      points[[i]]$cluster <- currentCluster
      findClusterPoints(x, currentCluster, points, i, eps, MinPts)
    }
  }
  
  end_time <- Sys.time() # Fin du chronométrage
  cat("Temps d'exécution DBSCAN:", end_time - start_time, "secondes\n")
  
  return(points)
}

# Génération de données synthétiques pour les clients
# Complexité : O(k × n) - Avec k le nombre de clusters et n le nombre de points
centers <- list(c(1, 1), c(-1, -1), c(1, -1), c(-1, 1))
n_samples <- 1000
cluster_std <- 0.4
set.seed(0)

X <- matrix(nrow = 0, ncol = 2)
labels_true <- c()

for (i in seq_along(centers)) {
  cluster_data <- mvrnorm(n_samples / length(centers), mu = centers[[i]], Sigma = diag(cluster_std^2, 2))
  X <- rbind(X, cluster_data)
  labels_true <- c(labels_true, rep(i, n_samples / length(centers)))
}

# Standardisation des données
# Complexité : O(n)
scaled_data <- scale(X)
scaled_df <- data.frame(X1 = scaled_data[, 1], X2 = scaled_data[, 2], labels = as.factor(labels_true))
colnames(scaled_df) <- c("x", "y")

# Visualisation des données standardisées
cat("Visualisation des données standardisées\n")
ggplot(scaled_df, aes(x = x, y = y)) +
  geom_point() +
  theme_minimal() +
  labs(
    title = "Nuage de points des clients en fonction de leurs fréquences d'achat et visites",
    x = "Fréquence d'achat (normalisée)",
    y = "Fréquence de visite (normalisée)"
  )

# Création des clusters
cat("Clustering en cours...\n")
points <- DBscan(scaled_data, eps = 0.2, MinPts = 5)

# Création d'un dataframe pour la visualisation
# Complexité : O(n)
clusters <- numeric()
clusters <- unlist(lapply(points, function(point) point$cluster))

# Convertir les clusters en facteur pour éviter les problèmes de coercion
scaled_df$clusters <- factor(
  ifelse(clusters == -1, "Outliers",
         ifelse(clusters == -2, "Border points", as.character(clusters)))
)

# Calcul de l'ARI (Adjusted Rand Index)
# Complexité : O(n)
adjusted_rand_index <- adjustedRandIndex(as.numeric(labels_true), as.numeric(as.character(clusters)))
cat("Adjusted Rand Index (ARI):", adjusted_rand_index, "\n")

# Visualisation des clusters
# Complexité : O(n)
cat("Visualisation des clusters\n")
ggplot(data = scaled_df, aes(x = x, y = y, color = clusters)) +
  geom_point() +
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 8)
  ) +
  scale_color_manual(values = c("green", "red", "purple", "brown", "blue", "orange"), name = "Clusters")




