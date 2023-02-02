import numpy as np
import math
from copy import deepcopy
from typing import List

class HeapNode(object):
    def __init__(self,
                 data,
                 heap_index=-1,
                 in_heap=False,
                 ):
        self.data = data
        self.heap_index = heap_index
        self.in_heap = in_heap

def key_default(node):
    return node.data

def less_than_default(a, b):
    return a.data < b.data

def greater_than_default(a, b):
    return a.data > b.data

def mark_default(node):
    node.in_heap = True
    
def unmark_default(node):
    node.in_heap = False
    
def marked_default(node):
    return node.in_heap

def set_index_default(node, val):
    node.heap_index = val
    
def unset_index_default(node):
    node.heap_index = -1
    
def get_index_default(node):
    return node.heap_index
 
class BinaryHeap(object):
    def __init__(self,
                 max_size=None,
                 key=key_default,
                 less_than=less_than_default,
                 greater_than=greater_than_default,
                 mark=mark_default,
                 unmark=unmark_default,
                 marked=marked_default,
                 set_index=set_index_default,
                 unset_index=unset_index_default,
                 get_index=get_index_default):

        if max_size is None:
            self.max_size = 64
        else:
            self.max_size = max_size
        
        self.heap_node: List[HeapNode] = []
        self.index_of_last = -1
        self.parent_of_last = -1
        self.key = key
        self.less_than = less_than
        self.greater_than = greater_than
        self.mark = mark
        self.unmark = unmark
        self.marked = marked
        self.set_index = set_index
        self.unset_index = unset_index
        self.get_index = get_index
        
def bubble_up(H: BinaryHeap, n: int):
    if (n==0):
        return
    if n > H.index_of_last:
        return

    parent = (n-1)//2
    while n != 0 and H.greater_than(H.heap_node[parent], H.heap_node[n]):
        temp_node = H.heap_node[parent]
        H.heap_node[parent] = H.heap_node[n]
        H.heap_node[n] = temp_node
        
        H.set_index(H.heap_node[parent], parent)
        H.set_index(H.heap_node[n], n)
        
        n = parent
        parent = (n-1)//2
    
def bubble_down(H: BinaryHeap, n: int):
    if 2*n+1 == H.index_of_last:
        child = 2*n+1
    elif 2*n+2 > H.index_of_last:
        return
    elif H.less_than(H.heap_node[2*n+1], H.heap_node[2*n+2]):
        child = 2*n+1
    else:
        child = 2*n+2
    
    while n <= H.parent_of_last and H.less_than(H.heap_node[child], H.heap_node[n]):
        temp_node = H.heap_node[child]
        H.heap_node[child] = H.heap_node[n]
        H.heap_node[n] = temp_node
        
        H.set_index(H.heap_node[child], child)
        H.set_index(H.heap_node[n], n)
        
        n = child
        
        if 2*n+1 == H.index_of_last:
            child = 2*n+1
        elif 2*n+2 > H.index_of_last:
            return
        elif H.less_than(H.heap_node[2*n+1], H.heap_node[2*n+2]):
            child = 2*n+1
        else:
            child = 2*n+2
            
def add_to_heap(H: BinaryHeap, this_node):
    
    if not H.marked(this_node):
        
        H.heap_node.append(this_node)
        H.index_of_last = len(H.heap_node) - 1
        H.parent_of_last = (H.index_of_last-1)//2
        H.set_index(this_node, H.index_of_last)
        bubble_up(H, H.index_of_last)
        
        H.mark(this_node)
        
        if len(H.heap_node) > H.max_size:
            remove_from_heap(H, H.heap_node[-1])
        
    else:
        raise ValueError("problems")
    
def top_heap(H: BinaryHeap):
    if H.index_of_last < 0:
        return False
    
    return H.heap_node[0]

def pop_heap(H: BinaryHeap):    
    if H.index_of_last < 0:
        return False
    
    old_top_node = H.heap_node[0]
    H.heap_node[0] = H.heap_node[H.index_of_last]
    H.set_index(H.heap_node[0], 0)
    H.heap_node.pop(H.index_of_last)
    H.index_of_last = len(H.heap_node) - 1
    H.parent_of_last = (H.index_of_last-1)//2
    bubble_down(H, 0)
    H.unmark(old_top_node)
    H.unset_index(old_top_node)
    return old_top_node

def remove_from_heap(H: BinaryHeap, this_node):
    n = H.get_index(this_node)
    
    moved_node = H.heap_node[H.index_of_last]
    H.heap_node[n] = moved_node
    H.heap_node.pop(H.index_of_last)
    H.set_index(moved_node, n)
    H.index_of_last = len(H.heap_node) - 1
    H.parent_of_last = (H.index_of_last-1)//2
    bubble_up(H, n)
    bubble_down(H, H.get_index(moved_node))
    H.unmark(this_node)
    H.unset_index(this_node)
    
def update_heap(H: BinaryHeap, this_node):
    if not H.marked(this_node):
        raise ValueError("trying to update a node that is not in the heap")
    bubble_up(H, H.get_index(this_node))
    bubble_down(H, H.get_index(this_node))
    
def print_heap(H: BinaryHeap):
    i = 0
    p = 0
    
    while i <= H.index_of_last:
        print(H.key(H.heap_node[i]), " ->   ")
        if 2*i <= H.index_of_last and i != 0:
            print(H.key(H.heap_node[2*i]))
        else:
            print("NULL")
        
        print("     ")
        
        if 2*i+1 <= H.index_of_last:
            print(H.key(H.heap_node[2*i+1]))
        else:
            print("NULL")
        
        i+=1
        
def print_pop_all_heap(H: BinaryHeap):
    while H.index_of_last >= 0:
        node = pop_heap(H)
        print(H.key(node))

def check_heap(H: BinaryHeap):
    i = 1
    if H.index_of_last < 0:
        print("Heap is empty")
        return True
    elif H.get_index(H.heap_node[0]) != 0:
        print("There is a problem with the heap (root)")
        return False

    while i <= H.index_of_last:
        if (H.less_than(H.heap_node[i], H.heap_node[(i-1)//2])):
            print("There is a problem with the heap order")
            return False
        elif (H.get_index(H.heap_node[i]) != i):
            print("There is a problem with the heap node data ", H.get_index(H.heap_node[i]), " != ", i)
            return False
        i+=1
    
    print("The heap is OK")
    return True

def clean_heap(H: BinaryHeap):
    ret_nodes = H.heap_node
    for i in range(H.index_of_last + 1):
        H.unmark(H.heap_node[i])
        H.unset_index(H.heap_node[i])
    H.heap_node.clear()
    H.index_of_last = -1
    H.parent_of_last = -1
    
    return ret_nodes
    
def bubble_up_b(H: BinaryHeap, n: int):
    if (n==0):
        return
    if n > H.index_of_last:
        return

    parent = (n-1)//2
    while n != 0 and H.less_than(H.heap_node[parent], H.heap_node[n]):
        temp_node = H.heap_node[parent]
        H.heap_node[parent] = H.heap_node[n]
        H.heap_node[n] = temp_node
        
        H.set_index(H.heap_node[parent], parent)
        H.set_index(H.heap_node[n], n)
        
        n = parent
        parent = (n-1)//2
        
def bubble_down_b(H: BinaryHeap, n: int):
    if 2*n+1 == H.index_of_last:
        child = 2*n+1
    elif 2*n+2 > H.index_of_last:
        return
    elif H.greater_than(H.heap_node[2*n+1], H.heap_node[2*n+2]):
        child = 2*n+1
    else:
        child = 2*n+2
    
    while n <= H.parent_of_last and H.greater_than(H.heap_node[child], H.heap_node[n]):
        temp_node = H.heap_node[child]
        H.heap_node[child] = H.heap_node[n]
        H.heap_node[n] = temp_node
        
        H.set_index(H.heap_node[child], child)
        H.set_index(H.heap_node[n], n)
        
        n = child
        
        if 2*n+1 == H.index_of_last:
            child = 2*n+1
        elif 2*n+2 > H.index_of_last:
            return
        elif H.greater_than(H.heap_node[2*n+1], H.heap_node[2*n+2]):
            child = 2*n+1
        else:
            child = 2*n+2
    
def add_to_heap_b(H: BinaryHeap, this_node):
    if not H.marked(this_node):
        H.heap_node.append(this_node)
        H.index_of_last = len(H.heap_node) - 1
        H.parent_of_last = (H.index_of_last-1)//2
        H.set_index(this_node, H.index_of_last)
        bubble_up_b(H, H.index_of_last)
        H.mark(this_node)
        
        if len(H.heap_node) > H.max_size:
            remove_from_heap_b(H, H.heap_node[-1])
    else:
        raise ValueError("problems")
    
def top_heap_b(H: BinaryHeap):
    return top_heap(H)

def pop_heap_b(H: BinaryHeap):
    if H.index_of_last < 0:
        return False
    
    old_top_node = H.heap_node[0]
    H.heap_node[0] = H.heap_node[H.index_of_last]
    H.set_index(H.heap_node[0], 0)
    H.heap_node.pop(H.index_of_last)
    H.index_of_last = len(H.heap_node) - 1
    H.parent_of_last = (H.index_of_last-1)//2
    bubble_down_b(H, 0)
    H.unmark(old_top_node)
    H.unset_index(old_top_node)
    return old_top_node

def remove_from_heap_b(H: BinaryHeap, this_node):
    n = H.get_index(this_node)
    
    moved_node = H.heap_node[H.index_of_last]
    H.heap_node[n] = moved_node
    H.heap_node.pop(H.index_of_last)
    H.set_index(moved_node, n)
    H.index_of_last = len(H.heap_node) - 1
    H.parent_of_last = (H.index_of_last-1)//2
    bubble_up_b(H, n)
    bubble_down_b(H, H.get_index(moved_node))
    H.unmark(this_node)
    H.unset_index(this_node)
    
def update_heap_b(H: BinaryHeap, this_node):
    if not H.marked(this_node):
        raise ValueError("trying to update a node that is not in the heap\n")
    bubble_up_b(H, H.get_index(this_node))
    bubble_down_b(H, H.get_index(this_node))
    
def print_heap_b(H: BinaryHeap):
    print_heap(H)
        
def print_pop_all_heap_b(H: BinaryHeap):
    print_pop_all_heap(H)

def check_heap_b(H: BinaryHeap):
    i = 1
    if H.index_of_last < 0:
        print("Heap is empty")
        return True
    elif H.get_index(H.heap_node[0]) != 0:
        print("There is a problem with the heap (root)")
        return False

    while i<= H.index_of_last:
        if (H.greater_than(H.heap_node[i], H.heap_node[(i-1)//2])):
            print("There is a problem with the heap order")
            return False
        elif (H.get_index(H.heap_node[i]) != i):
            print("There is a problem with the heap node data ", H.get_index(H.heap_node[i]), " != ", i)
            return False
        i+=1
    
    print("The heap is OK")
    return True

def clean_heap_b(H: BinaryHeap):
    clean_heap(H)
     
