import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Point:
    def __init__(self, coordinates, type, neighboringPoints, cluster):
        self.coordinates = coordinates
        self.type = type    # 1:core point, 2:border point, 3:outlier
        self.neighboringPoints = neighboringPoints
        self.cluster = cluster    # 1, 2, 3, 4, ... name of cluster

def EuclDist(x, p, eps, MinPts): # step:1
    """
        classification of each points  (cores contient plus de Minpts pts ,borders entre 2 et Minpts,outlier ne contient aucun pts)
        return [indexes of the neighbors of p, type of p]
    """
    NeighboringPoints = []
    type = 0
    for j in range(len(x)):
        point = x[j]
        dist = np.linalg.norm(np.array(point) - np.array(p))
        if dist < eps:
            NeighboringPoints.append(j)
    if len(NeighboringPoints) > MinPts:
        type = 1  # Core point
    elif len(NeighboringPoints) > 1:
        type = 2  # Border point
    else:
        type = 3  # Outlier
    return [NeighboringPoints, type]


def findClusterPoints(x, currentCluster, points, position, eps, MinPts): # step:2
    ClusterMembers = points[position].neighboringPoints
    i = 0
    while (i < len(ClusterMembers)):
        expansionPoint = ClusterMembers[i] #set an expansion point
        if (points[expansionPoint].cluster == -1): #if it's a border point that has not been assigned to a cluster
            points[expansionPoint].cluster = currentCluster
        elif (points[expansionPoint].cluster == 0): #if it's a core point that has not been assigned to a cluster
            points[expansionPoint].cluster = currentCluster
            ClusterMembers = ClusterMembers + points[expansionPoint].neighboringPoints
        i = i + 1

def DBscan(x, eps, MinPts):     # x : the coordinates of the datapoints
    currentCluster = 0
    points = []
    for data in x:      # initialize points as core, border, or outlier points
        [NeighboringPoints, type] = EuclDist(x, data, eps, MinPts)
        points.append(Point(data, type, NeighboringPoints, 1 - type))
        # For core points (type = 1): 1 - type = 1−1=0
        # For border points (type = 2): 1 - type =1−2=−1
        # For outliers (type = 3): 1 - type = 1−3 =−2

    for i in range(len(x)): # loop through all datapoints
        if points[i].cluster == 0: # if a point is a core point and it has not been clustered yet
            currentCluster = currentCluster + 1
            points[i].cluster = currentCluster
            findClusterPoints(x, currentCluster, points, i, eps, MinPts)
    return points



########################################################################################################################################

from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# random data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs( n_samples=1000, centers=centers, cluster_std=0.4, random_state=0 )
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ploting the brute data
plt.scatter(X[:,0], X[:,1])
plt.title("Random dataset")
plt.xlabel("x")
plt.ylabel("y") 
plt.show()


# DBSCAN
points = DBscan(X, eps = 0.2, MinPts = 5)
clusters = []
for point in points:
    clusters.append(point.cluster)
    df = pd.DataFrame(X, columns = ["x", "y"])
df['clusters'] = np.array(clusters)
df = df.sort_values("clusters")
df['clusters'] = df['clusters'].apply(lambda x: 'Outliers' if x == -2 else x)
df['clusters'] = df['clusters'].apply(lambda x: "Border points that don't belong to clusters" if x == -1 else x)


# plot the clustered data 
plt.figure(figsize=(8, 6))
sns.scatterplot(data = df, x = "x", y = "y", hue = "clusters")
plt.title("Clustred Dara (after using DBSCAN)")
plt.xlabel("x")
plt.ylabel("y") 
plt.legend(loc='upper right', fontsize='x-small')
plt.show()