# Author: Yiming Zhang
from unittest import result
import numpy as np

# Read Files
def Readfile(File):
    data=[]
    with open(File, 'r') as f:
        for x in f.readlines():
            x=x.strip('\n')
            x=x.split(' ')
            data.append(x)
    return data

# Merge Sort
def mergeSort(List):
    if len(List) > 1:
        mid = len(List) // 2
        left = List[:mid]
        right = List[mid:]

        mergeSort(left)
        mergeSort(right)

        i = 0
        j = 0
        k = 0
        
        while i < len(left) and j < len(right):
            if left[i].data <= right[j].data:
                List[k] = left[i]
                i += 1
            else:
                List[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            List[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            List[k]=right[j]
            j += 1
            k += 1

def countADP(list, NumRef):
    count = []
    for i in range(NumRef-1):
        Ci = 0
        for j in range(len(list)):
            if list[j] == i:
                Ci += 1
        count.append(Ci/len(list))
    count.append(1-sum(count))
    return count

def ResultAnalysis(list, number, NumRef):
    ADP = []
    for i in range(number):
        ADP_sub = []
        for j in range(2**i):
            lists = list[int(j/(2**i)*len(list)):int((j+1)/(2**i)*len(list))]
            ADP_sub.append(countADP(lists, NumRef))
        ADP.append(ADP_sub)
    return ADP


