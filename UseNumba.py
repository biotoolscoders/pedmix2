# Author: Yiming Zhang
import numpy as np
from numba import jit
from numba.experimental import jitclass
from numba import njit
from numba import int32, deferred_type, optional, types, typeof, typed
from collections import OrderedDict


# Define classes 

# Tree Nodes class
node_type = deferred_type()

spec = OrderedDict()
spec['data'] = int32
spec['lchild'] = optional(node_type)
spec['rchild'] = optional(node_type)
spec['ancestry'] = optional(node_type)


@jitclass(spec)
class TreeNode(object):
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.ancestry = None


node_type.define(TreeNode.class_type.instance_type)

stack_type = deferred_type()

spec = OrderedDict()
spec['data'] = TreeNode.class_type.instance_type
spec['next'] = optional(stack_type)


# Stack

@jitclass(spec)
class Stack(object):
    def __init__(self, data, next):
        self.data = data
        self.next = next

stack_type.define(Stack.class_type.instance_type)

@njit
def push(stack, data):
    return Stack(data, stack)

@njit
def pop(stack):
    return stack.next

@njit
def make_stack(data):
    return push(None, data)

# Pre-Order Traversal
@njit
def list_preorder(node):
    out = []
    stack = make_stack(node)

    while stack is not None:
        node = stack.data
        out.append(node)
        stack = pop(stack)

        if node.rchild is not None:
            stack = push(stack, node.rchild)
        if node.lchild is not None:
            stack = push(stack, node.lchild)

    return out
    

# Creat Pedigree    
@njit
def creat(root, llist, i, j):
    if i < len(llist):
        root = TreeNode(llist[i])
        root.ancestry = j
        root.rchild = creat(root.rchild, llist, 2*i+1, root)
        root.lchild = creat(root.lchild, llist, 2*i+2, root)
        return root
    return root

# Log Add
@jit(nopython=True)
def logA(a,b):
    if(a>b):
        c = a + np.log(1.0+np.exp(b-a))
    else:
        c = b + np.log(1.0+np.exp(a-b))
    return c

# Comparsion of size of two strings 
@jit(nopython=True)
def CompareNP(S1, S2):
    for i in range(len(S1)):
        if S1[i] > S2[i]:
            return True
        elif S1[i] < S2[i]:
            return False
        else:
            continue
    return False        


# SMA algorithm
@jit(nopython=True)
def RNF(s):
    if len(s) == 1:
        return s
    else:
        k = len(s)/2
        S1 = RNF(s[0:int(k)])
        S2 = RNF(s[int(k):len(s)])
        if CompareNP(S1, S2):
            return np.concatenate((S2, S1))
        else:
            return np.concatenate((S1, S2))


# Transform the array to string
@jit(nopython=True)
def ArraytoString(S):
    S1 = S.astype('int32')
    S2 = ''
    for i in range(len(S1)):
        S2 += str(S1[i])
    return S2