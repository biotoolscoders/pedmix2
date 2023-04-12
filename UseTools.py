# Author: Yiming Zhang
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