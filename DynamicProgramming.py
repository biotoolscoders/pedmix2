# Author: Yiming Zhang
import numpy as np
from UseNumba import *


@jit(nopython=True)
def Calc_Prob_fast(Founder_List, Ref, length, PNoS, PS, NodeList, Num_generations):
    log5 = np.log(0.5)
    DPTable = np.zeros((2**(Num_generations+1)-1, 2, length, length))
    for i in range(0, (2**(Num_generations-1))):
        for j in range(2):
            DPTable[i*2+j,0,:,:] = Ref[int(Founder_List[i])]
            DPTable[i*2+j,1,:,:] = Ref[int(Founder_List[i])]
            
    for i in range((2**Num_generations), NodeList[-1].data + 1):
        for s in range(length - 1, -1, -1):
            for k in range(s + 1, length + 1):
                Prob1 = np.iinfo(np.int32).min
                Prob2 = np.iinfo(np.int32).min
                for j in range(s , k):
                    if j == k - 1:
                        Prob1 = logA(Prob1, (PNoS[s][j-s] + log5 + logA(DPTable[NodeList[i].lchild.data,0,s,k-s-1], DPTable[NodeList[i].lchild.data,1,s,k-s-1])))
                        Prob2 = logA(Prob2, (PNoS[s][j-s] + log5 + logA(DPTable[NodeList[i].rchild.data,0,s,k-s-1], DPTable[NodeList[i].rchild.data,1,s,k-s-1])))
                        
                        DPTable[i,0,s,j-s] = Prob1
                        DPTable[i,1,s,j-s] = Prob2
                    else:
                        Prob1 = logA(Prob1, (PS[s][j-s] + log5 + logA(DPTable[NodeList[i].lchild.data,0,s,j-s], DPTable[NodeList[i].lchild.data,1,s,j-s]) + DPTable[i,1,j+1,k-j-2]))
                        Prob2 = logA(Prob2, (PS[s][j-s] + log5 + logA(DPTable[NodeList[i].rchild.data,0,s,j-s], DPTable[NodeList[i].rchild.data,1,s,j-s]) + DPTable[i,0,j+1,k-j-2]))    
    return DPTable

@jit(nopython=True)
def Calc_Prob_Hill_Clim(list_id, list_p, DPtable, Ref, length, PNoS, PS, NodeList):
    log5 = np.log(0.5)
    DPtable[list_id*2, 0, :, :] = Ref[int(list_p)]
    DPtable[list_id*2, 1, :, :] = Ref[int(list_p)]
    DPtable[list_id*2+1, 0, :, :] = Ref[int(list_p)]
    DPtable[list_id*2+1, 1, :, :] = Ref[int(list_p)]
    next_node = NodeList[list_id*2].ancestry
    while next_node is not None:
        for s in range(length - 1, -1, -1):
            for k in range(s + 1, length + 1):
                Prob1 = np.iinfo(np.int32).min
                Prob2 = np.iinfo(np.int32).min
                for j in range(s , k):
                    if j == k - 1:
                        Prob1 = logA(Prob1, (PNoS[s][j-s] + log5 + logA(DPtable[next_node.lchild.data,0,s,k-s-1], DPtable[next_node.lchild.data,1,s,k-s-1])))
                        Prob2 = logA(Prob2, (PNoS[s][j-s] + log5 + logA(DPtable[next_node.rchild.data,0,s,k-s-1], DPtable[next_node.rchild.data,1,s,k-s-1])))
                        DPtable[next_node.data,0,s,j-s] = Prob1
                        DPtable[next_node.data,1,s,j-s] = Prob2
                    else:
                        Prob1 = logA(Prob1, (PS[s][j-s] + log5 + logA(DPtable[next_node.lchild.data,0,s,j-s], DPtable[next_node.lchild.data,1,s,j-s]) + DPtable[next_node.data,1,j+1,k-j-2]))
                        Prob2 = logA(Prob2, (PS[s][j-s] + log5 + logA(DPtable[next_node.rchild.data,0,s,j-s], DPtable[next_node.rchild.data,1,s,j-s]) + DPtable[next_node.data,0,j+1,k-j-2]))          
        next_node = next_node.ancestry