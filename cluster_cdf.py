import matplotlib.pyplot as plt
import numpy as np

#ref: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score
from sklearn.cluster import KMeans

def readFile2Input(fileData):
    fp = open(fileData, 'r', encoding="Latin-1")
    dicUser = {}  # userid:index
    totalUser = 0
    LocF = []
    while True:
        data = fp.readline()
        if len(data) == 0:
            break
        data = data.rstrip('\n')
        dat  = data.split('\t')

        usr = dat[0]
        loc = dat[1]
        if usr not in dicUser :
            #add usr into Dics
            if len(dicUser) == 0 :
                dicUser = {usr:totalUser}
            else :
                dicUser[usr] = totalUser
            totalUser+=1
            LocF.append({})
        idu = dicUser[usr]
        if len(LocF[idu]) == 0:
            LocF[idu] = {loc: 1}
        else:
            if loc in LocF[idu]:
                LocF[idu][loc] += 1
            else:
                LocF[idu][loc] = 1
    return dicUser, LocF

def input2CDF(LocF):
    UserFreq = []  # tatal frequency
    UserLocFreq = []  # total distinct lacation frequency
    for locs in LocF:
        UserLocFreq.append(len(locs))
        UserFreq.append(sum(list(locs.values())))
    return UserFreq, UserLocFreq

def CDF(UserFreq, UserLocFreq, thF=0.5, thL=0.5):
    # print(UserFreq, UserLocFreq)
    UserFreq_sorted = np.sort(UserFreq)
    UserLocFreq_sorted = np.sort(UserLocFreq)
    CDF_UserFreq = 1. * np.arange(len(UserFreq_sorted)) / (len(UserFreq_sorted) - 1)
    CDF_UserLocFreq = 1. * np.arange(len(UserLocFreq_sorted)) / (len(UserLocFreq_sorted) - 1)
    #-----------------------------------
    ufreq = -1
    uloc  = -1
    i=0
    for dat in CDF_UserFreq :
        if dat >= thF :
            ufreq =  UserFreq_sorted[i]
            break
        i+=1
    i=0
    for dat in CDF_UserLocFreq:
        if dat >= thL :
            # print(dat, UserLocFreq_sorted[i])
            uloc = UserLocFreq_sorted[i]
            break
        i+=1
    # plot the sorted data:
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1.plot(UserFreq_sorted, CDF_UserFreq)
    # ax1.set_xlabel('#Frequency of users')
    # ax1.set_ylabel('CDF')
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(UserLocFreq_sorted, CDF_UserLocFreq)
    # ax2.set_xlabel('#Location frequency of users')
    # ax2.set_ylabel('CDF')
    # plt.show()
    return ufreq, uloc

def createInputs(dicUser,LocF,ufreq, uloc):
    inputs = []
    active_u = {}
    total_active = 0
    for idu in range(0, len(LocF)) :
        rt = 0
        allfreq = 0
        if  len(LocF[idu]) >= uloc :
            for loc in LocF[idu] :
                if LocF[idu][loc] > 1 :
                    rt+=1
                allfreq += LocF[idu][loc]
            if allfreq >= ufreq :
                userid = list(dicUser.keys())[list(dicUser.values()).index(idu)]
                if len(active_u) == 0 :
                    active_u = {userid:total_active}
                else :
                    active_u[userid] = total_active
                inputs.append([0]*3)
                inputs[total_active][0] = allfreq
                inputs[total_active][1] = len(LocF[idu]) #LN
                inputs[total_active][2] = rt
                total_active += 1
    # print('inputs ',inputs)
    # print('active_u ',active_u)
    return np.array(inputs), active_u

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
    if 1 : #file inputs
        path = "C:/Users/HP PAVILION/PycharmProjects/LBSN/"
        fileLoc = path + "Bangsaen_Food_location.json"
        fileData = path + "Bangsaen-check-in.txt"
        # district_name = 'Bangsaen'
        # fileLoc = path + "FullTKY_Food_location.json"
        # fileData = path + "FullTKY_Food.txt"
        #
        fileData = "C:/Users/HP PAVILION/Desktop/Datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"

    dicUser, LocF = readFile2Input(fileData)
    UserFreq, UserLocFreq = input2CDF(LocF)
    ufreq, uloc = CDF(UserFreq, UserLocFreq, 0.4, 0.6)
    # print(dicUser)
    # print(LocF)
    # print('UserFreq    ',UserFreq, 'ufreq',ufreq)
    # print('UserLocFreq ',UserLocFreq, 'uloc',uloc)
    inputs, active_u = createInputs(dicUser,LocF,ufreq, uloc)
    k = 5
    Clustering(inputs, k)