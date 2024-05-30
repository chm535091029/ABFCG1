import radon
from radon.visitors import CFGVisitor
from radon.complexity import cc_rank

code = """
def my_function(a, b):
    if a > b:
        return a - b
    else:
        return b - a
"""

# 使用Radon的ast模块解析代码，生成抽象语法树
ast_node = radon.ast.parse(code)

# 使用Radon的visitors模块中的CFGVisitor类来分析抽象语法树，从而得到控制流图
cfg = CFGVisitor.from_ast(ast_node)

# 遍历控制流图的节点，计算每个节点的圈复杂度，并将其保存到字典中
cc_dict = {}
for node in cfg.nodes:
    cc_dict[node] = cc_rank(node)

print(cc_dict)
