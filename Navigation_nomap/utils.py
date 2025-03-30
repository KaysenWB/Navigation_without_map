import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt


class Pos_Emb(nn.Module):
    def __init__(self):
        super().__init__()
        self.landmarks = np.load('./data/landmark.npy')
        self.emb_coe = 10

    def forward(self, node):
        diff = np.sqrt((node - self.landmarks)**2)
        #dis = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
        #Emb = np.exp(diff * -self.emb_coe)
        #Emb = Emb/np.sum(Emb)

        return diff

class Pos_Emb2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.landmarks = np.load(args.data_root + '/landmark.npy')
        self.emb_coe = 1

    def forward(self, node):
        diff = node - self.landmarks
        dis = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
        Emb = np.exp(dis * -self.emb_coe)
        Emb = Emb/np.sum(Emb)

        return Emb

def Create_landmark(min, max, nodes):
    landmark = np.zeros((nodes,2))
    landmark[:,0] = np.random.uniform(min, max,nodes)
    landmark[:,1] = np.random.uniform(min, max,nodes)
    np.save("./data/landmark.npy", landmark)

    test = np.load('./data/landmark.npy')
    plt.scatter(test[:, 0], test[:, 1])
    plt.show()
    return

