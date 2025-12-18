from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

EXAMPLE_MARKDOWN = """
## Heading2 here

### Heading3 here
Some paragraph text and **emphasis here** and more text here.
"""


def main():
    tokens = MarkdownIt().parse(EXAMPLE_MARKDOWN)
    tree = SyntaxTreeNode(tokens)
    for node in tree.walk():
        if node.is_root:
            print(node.type)
        elif node.type == "heading":
            print(f"Type: {node.type}, tag: {node.tag}, level: {node.level}, markup: {node.markup}")
        else:
            print(f"{node.type}: {node.content}")


if __name__ == "__main__":
    main()
