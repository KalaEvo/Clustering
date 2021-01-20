#ref: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
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

def calculate_Calinski_Harabasz0(x, labels):
    #sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    return calinski_harabasz_score(x, labels)

def calculate_Davies_Bouldin0(x, labels):
    #sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    return davies_bouldin_score(x, labels)

def readFile2Input():
    path = "C:/Users/HP PAVILION/PycharmProjects/LBSN/"
    fileLoc = path + "Bangsaen_Food_location.json"
    fileData = path + "Bangsaen-check-in.txt"
    # district_name = 'Bangsaen'

    # fileLoc = path + "FullTKY_Food_location.json"
    # fileData = path + "FullTKY_Food.txt"

    numFeatures = 3 #0 : total frequency 1 : distint locations 2: time distribution
    dicUser = {} #userid:index
    totalUser = 0
    inputs = [] #matrix row : users col : features
    TD = [] ##matrix row : users col: time 0-23
    LN = [] #matrix row : users col: distinct locations

    fp = open(fileData, 'r', encoding="UTF-8-sig")
    while True:
        data = fp.readline()
        if len(data) == 0:
            break
        data = data.rstrip('\n')
        dat  = data.split('\t')
        # print(dat)
        usr  = dat[0]
        loc  = dat[1]
        hr   = int(dat[7][11:13])
        if usr not in dicUser :
            #add usr into Dics
            if len(dicUser) == 0 :
                dicUser = {usr:totalUser}
            else :
                dicUser[usr] = totalUser
            totalUser+=1
            TD.append([0]*24)
            LN.append([])
            inputs.append([0]*numFeatures)
        idu = dicUser[usr]
        inputs[idu][0]+=1
        if loc not in LN[idu] :
            LN[idu].append(loc)

        TD[idu][hr]=1
    idu=0
    for locs in LN :
        inputs[idu][1] = len(locs)
        idu+=1
    idu = 0
    for hrs in TD :
        inputs[idu][2] = sum(hrs)
        idu+=1
    print('inputs',inputs)
    # print('TD',TD)
    # print('LN',LN)
    print('totalUser', totalUser )
    return np.array(inputs)

def readFile2Input_5Features():
    path = "C:/Users/HP PAVILION/PycharmProjects/LBSN/"
    fileLoc = path + "Bangsaen_Food_location.json"
    fileData = path + "Bangsaen-check-in.txt"
    # district_name = 'Bangsaen'

    # fileLoc = path + "FullTKY_Food_location.json"
    # fileData = path + "FullTKY_Food.txt"
    #
    # fileData = "C:/Users/HP PAVILION/Desktop/Datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"

    numFeatures = 4
    #0 : total frequency 1 : distint locations 2: time distribution
    # 3:Revisit
    dicUser = {} #userid:index
    totalUser = 0
    inputs = [] #matrix row : users col : features
    LocF = []

    TD = [] ##matrix row : users col: time 0-23
    fp = open(fileData, 'r', encoding="UTF-8-sig")
    # fp = open(fileData, 'r', encoding="Latin-1") #ใช้กับdata set ใหญ่
    while True:
        data = fp.readline()
        if len(data) == 0:
            break
        data = data.rstrip('\n')
        dat  = data.split('\t')
        # print(dat)
        usr  = dat[0]
        loc  = dat[1]
        hr   = int(dat[7][11:13])
        if usr not in dicUser :
            #add usr into Dics
            if len(dicUser) == 0 :
                dicUser = {usr:totalUser}
            else :
                dicUser[usr] = totalUser
            totalUser+=1
            TD.append([0]*24)
            LocF.append({})
            inputs.append([0]*numFeatures)

        idu = dicUser[usr]
        if len(LocF[idu])==0:
            LocF[idu] = {loc:1}
        else :
            if loc in LocF[idu] :
                LocF[idu][loc]+=1
            else :
                LocF[idu][loc] = 1
        TD[idu][hr]=1

    for idu in range(0, len(LocF)) :
        rt = 0
        allfreq = 0
        for loc in LocF[idu] :
            if LocF[idu][loc] > 1 :
                rt+=1
            allfreq += LocF[idu][loc]
        # print(idu, allfreq, rt, len(LocF[idu]))
        inputs[idu][0] = allfreq
        inputs[idu][1] = len(LocF[idu]) #LN
        inputs[idu][3] = rt

    idu = 0
    for hrs in TD :
        inputs[idu][2] = sum(hrs)
        idu+=1
    # print('inputs',inputs)
    # print('LocF ',LocF)
    # print('TD',TD)
    # print('LN',LN)
    print('totalUser', totalUser)
    return np.array(inputs)


def Clustering(inputUserFeatures, kmax):
    sse = []
    sil = []
    db = []
    ch = []

    # Create dataset with 3 random cluster centers and 1000 datapoints
    # x, y = make_blobs(n_samples = 10, centers = 4, n_features=3, shuffle=True, random_state=31)
    # x = np.array([[1,2,3],[1,2,2.5],[1,3,2.5], [50,52,52.5],[58,52,52.5],[50,52,52.5], [100,200,150], [150,220,180]])
    # print(type(x))
    # print('data ', x)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit(inputUserFeatures)
      labels = kmeans.labels_
      # print('labels ',labels)
      centroids = kmeans.cluster_centers_
      pred_clusters = kmeans.predict(inputUserFeatures)

      ch.append(calculate_Calinski_Harabasz0(inputUserFeatures, labels))
      db.append(calculate_Davies_Bouldin0(inputUserFeatures, labels))
      sil.append(calculate_Silhouette0(inputUserFeatures, labels))
      sse.append(calculate_WSS0(inputUserFeatures, centroids, pred_clusters))

    print('Calinski_score ', ch)
    print('Davies_score ', db)
    print('Silhouette_score ', sil)
    print('Elbow_score ', sse)

if __name__ == '__main__':
    # inputUserFeatures = readFile2Input()
    inputUserFeatures = readFile2Input_5Features()
    Clustering(inputUserFeatures, 5)
    # sse = []
    # sil = []
    # kmax = 5
    #
    # # Create dataset with 3 random cluster centers and 1000 datapoints
    # x, y = make_blobs(n_samples = 10, centers = 4, n_features=3, shuffle=True, random_state=31)
    # x = np.array([[1,2,3],[1,2,2.5],[1,3,2.5], [50,52,52.5],[58,52,52.5],[50,52,52.5], [100,200,150], [150,220,180]])
    # print(type(x))
    # print('data ', x)
    # # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    # for k in range(2, kmax+1):
    #   kmeans = KMeans(n_clusters = k).fit(x)
    #   labels = kmeans.labels_
    #   centroids = kmeans.cluster_centers_
    #   pred_clusters = kmeans.predict(x)
    #
    #   sil.append(calculate_Silhouette0(x, labels))
    #   sse.append(calculate_WSS0(x, centroids, pred_clusters))
    #
    # print('Silhouette_score ', sil)
    # print('Elbow_score ', sse)