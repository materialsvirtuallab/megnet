from monty.pprint import pprint_table, draw_tree

import unittest


class PprintTableTest(unittest.TestCase):
    def test_print(self):
        table = [["one", "two"], ["1", "2"]]
        pprint_table(table)


class DrawTreeTest(unittest.TestCase):
    def test_draw_tree(self):
        class Node:
            def __init__(self, name, children):
                self.name = name
                self.children = children

            def __str__(self):
                return self.name

        root = Node(
            "root",
            [
                Node("sub1", []),
                Node("sub2", [Node("sub2sub1", [])]),
                Node(
                    "sub3",
                    [
                        Node("sub3sub1", [Node("sub3sub1sub1", [])]),
                        Node("sub3sub2", []),
                    ],
                ),
            ],
        )

        print(draw_tree(root))


if __name__ == "__main__":
    unittest.main()
