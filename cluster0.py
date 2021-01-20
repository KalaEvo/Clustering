from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# The Elbow Method function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

def calculate_WSS0(x, centroids, pred_clusters):
    curr_sse = 0
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(x)):
        curr_center = centroids[pred_clusters[i]]
        curr_sse += (x[i, 0] - curr_center[0]) ** 2 + (x[i, 1] - curr_center[1]) ** 2
    return curr_sse

def calculate_Silhouette0(x, labels):
    #sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    return silhouette_score(x, labels, metric='euclidean')


sse = []
sil = []
kmax = 4

# Create dataset with 3 random cluster centers and 1000 datapoints
x, y = make_blobs(n_samples = 10, centers = 4, n_features=3, shuffle=True, random_state=31)
print(type(x))
print('data ', x)
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(x)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_
  pred_clusters = kmeans.predict(x)

  sil.append(calculate_Silhouette0(x, labels))
  sse.append(calculate_WSS0(x, centroids, pred_clusters))

print('Silhouette_score ', sil)
print('Elbow_score ', sse)

