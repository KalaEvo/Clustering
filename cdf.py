import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def readFile2Input():
    path = "C:/Users/HP PAVILION/PycharmProjects/LBSN/"
    fileLoc = path + "Bangsaen_Food_location.json"
    fileData = path + "Bangsaen-check-in.txt"
    # district_name = 'Bangsaen'

    # fileLoc = path + "FullTKY_Food_location.json"
    # fileData = path + "FullTKY_Food.txt"
    #
    fileData = "C:/Users/HP PAVILION/Desktop/Datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"

    # fp = open(fileData, 'r', encoding="UTF-8-sig")
    fp = open(fileData, 'r', encoding="Latin-1") #ใช้กับdata set ใหญ่
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
    # print('LocF',LocF)
    # print('dicUser', dicUser)
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
            print(dat, UserFreq_sorted[i])
            ufreq =  UserFreq_sorted[i]
            break
        i+=1
    print('----------------')
    i=0
    for dat in CDF_UserLocFreq:
        if dat >= thL :
            print(dat, UserLocFreq_sorted[i])
            uloc = UserLocFreq_sorted[i]
            break
        i+=1
    print('----------------')
    # exit(1)
    #-----------------------------------
    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(UserFreq_sorted, CDF_UserFreq)
    ax1.set_xlabel('#Frequency of users')
    ax1.set_ylabel('CDF')

    ax2 = fig.add_subplot(122)
    ax2.plot(UserLocFreq_sorted, CDF_UserLocFreq)
    ax2.set_xlabel('#Location frequency of users')
    ax2.set_ylabel('CDF')
    plt.show()
    return ufreq, uloc

if __name__ == '__main__':
    dicUser, LocF = readFile2Input()
    UserFreq, UserLocFreq = input2CDF(LocF)
    # UserFreq = [1,1,1,1,2,3,4,5]
    # UserLocFreq = [1,1,1,1,2,3,4,5]
    ufreq, uloc = CDF(UserFreq, UserLocFreq, 0.6, 0.6)
    print('ufreq ', ufreq)
    print('uloc ', uloc)

    exit(1)
    '''
    # create some randomly ddistributed data:
    data = np.random.randn(1000)
    data = [1,1,1,1,2,3,4,5]
    # sort the data:
    data_sorted = np.sort(data)
    
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data)-1 )
    print(p)
    print(np.arange(2) / (7))
    norm_cdf = stats.norm.cdf(data)
    
    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(data_sorted, norm_cdf)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$p$')
    
    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')
    plt.show()
    '''