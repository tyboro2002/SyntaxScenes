from bst import BST
from avl import AVLTree
from red_black_tree import RedBlackTree

from trees.narry_tree import NaryTree

values = [5, 3, 7, 2, 4, 6, 8, 9, 10, 11, 12]

if __name__ == "__main__":
    # # For AVL Tree:
    # avl_tree = AVLTree()
    # avl_tree.animate_insertions(values)
    # print("AVL done")
    #
    # # For normal BST:
    # bst = BST()
    # bst.animate_insertions(values)
    # print("BST done")
    #
    # # For Red-Black Tree:
    # rb_tree = RedBlackTree()
    # rb_tree.animate_insertions(values)
    # print("Red-Black Tree done")

    # Create a new N-ary tree (for a 2-3 tree, max_children would be 3)
    tree = NaryTree(3)

    # Insert nodes with multiple values
    # tree.insert(5)
    # tree.insert(3)
    # tree.insert(7)
    # tree.insert(2)
    # tree.insert(6)
    # tree.insert(10)
    # tree.insert(12)

    tree.animate_insertions(values)

    tree.traverse()

    # Capture the frame for visualization
    tree.capture_frame('nary_tree.png')
