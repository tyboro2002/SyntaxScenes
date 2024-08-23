from bst import BST
from avl import AVLTree

values = [5, 3, 7, 2, 4, 6, 8, 9, 10, 11, 12]

if __name__ == "__main__":
    # For AVL Tree:
    avl_tree = AVLTree()
    avl_tree.animate_insertions(values)

    # For normal BST:
    bst = BST()
    bst.animate_insertions(values)
