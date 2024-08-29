from bst import BST
from avl import AVLTree
from red_black_tree import RedBlackTree

from trees.b_tree import BTree
from trees.b_tree_sub_classes import TwoThreeTree, TwoThreeFourTree

values = [5, 3, 7, 2, 4, 6, 8, 9, 10, 11, 12]

if __name__ == "__main__":
    # For AVL Tree:
    avl_tree = AVLTree()
    avl_tree.animate_insertions(values)
    print("AVL done")

    # For normal BST:
    bst = BST()
    bst.animate_insertions(values)
    print("BST done")

    # For Red-Black Tree:
    rb_tree = RedBlackTree()
    rb_tree.animate_insertions(values)
    print("Red-Black Tree done")

    # Create a B-tree of order 3 (2-3 Tree)
    two_three_tree = TwoThreeTree()
    two_three_tree.animate_insertions(values)
    print("2-3 Tree done")

    # Create a B-tree of order 3 (2-3 Tree)
    two_three_tree = TwoThreeTree()
    two_three_tree.animate_insertions(values)
    print("2-3 Tree done")

    # Create a B-tree of order 4 (2-3-4 Tree)
    two_three_four_tree = TwoThreeFourTree()
    two_three_four_tree.animate_insertions(values)
    print("2-3-4 Tree done")

    # Create a B-tree of order 8
    b_tree = BTree(order=8)
    b_tree.animate_insertions(values+[12, 12, 12, 12, 12, 12, 12, 12])
    print("B Tree done")
