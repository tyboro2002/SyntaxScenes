from bst import BST
from avl import AVLTree
from red_black_tree import RedBlackTree

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
