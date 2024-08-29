from b_tree import BTree


class TwoThreeTree(BTree):
    def __init__(self):
        super().__init__(order=3, title=f'2-3 Tree Visualization')  # Initialize BTree with order 3


class TwoThreeFourTree(BTree):
    def __init__(self):
        super().__init__(order=4, title=f'2-3-4 Tree Visualization')  # Initialize BTree with order 4
