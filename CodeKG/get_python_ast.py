import importlib

# nodes = importlib.import_module('tests.nodes')
# convert_node = nodes.convert_node
# flatten = nodes.flatten

from nodes import convert_node, flatten
# from tests.template import Template
import ast
import json

values = [
    "name",# for type = "FunctionDef"
    "arg",
    "id", # for type = "Name"



]
children = [
    "body",
    "args", #参数
    "targets",# for Assign
    "value",# for Assign
    "iter",# for For
    "elts", # for Tuple
    "func", #for If

]

ban = [[],"None"]


def convert2list(tree):
    global nodelists
    dict={}
    dict['id']=len(nodelists)
    dict['type'] = tree['type']
    dict['children'] = []
    for child in node.children:
        dict['children'].append(child.id)
    dict['value'] = node.token

    return dict
    t = type(node)

    if t is str or t is int:
        return node

    if t is list:
        return [convert_node(child) for child in node]

    if node is None:
        # return "nil"
        return "None"

    tname = t.__qualname__ #和__name__相似，比__name__高级一点
    d = {"type": tname}

    if t not in nodes:
        return f"#<{tname}>"

    for name in nodes[t]:
        d[name] = convert_node(getattr(node, name))
    return d

def checknesteddic(dic):
    '''
    递归拿到嵌套的树，每递归一次就定义一个node，储存所有非none，[](所有非空的键值对)，当作value
    23/4/21，lmc
    :param dic:
    :return:
    '''
    global nodelists
    dictnode={}
    dictnode['id']=str(len(nodelists))
    dictnode['children'] = []
    nodelists.append(dictnode)
    for key,value in dic.items():
        if(key == "type"):
            dictnode["type"] = value
        if(value not in ban and type(value)!=int and key!="type" and type(value)!=list and type(value)!=dict):
            # print(value)
            dictnode["value"] = value
        if(type(value)==dict):

        # if (isinstance(list, type(key)) or isinstance(dict, type(key))):
            dictnode['children'].append(checknesteddic(value))
        if(type(value)==list):
            for _ in value:
                dictnode['children'].append(checknesteddic(_))
    nodelists[int(dictnode["id"])] = dictnode
    return dictnode["id"]





# if __name__ == '__main__':
#     file_path = 'data.txt'
#     with open(file_path,'r') as src:
#         code = src.read()
#         raw_nodes = ast.dump(ast.parse(code))
#         nodelists = []
#
#         tree = convert_node(ast.parse(code))
#         # 扁平化嵌套的AST节点
#         s = checknesteddic(tree)
#         # final = convert2list(tree)
#         # parer_code = flatten(code)
#         root_node = tree['body'][0]
#         # print("test")
#         for i in nodelists:
#             print(i)
#         pass
nodelists = []
def parse_ast_python(source_code):
    global nodelists
    # 清空nodelists
    nodelists = []
    tree = convert_node(ast.parse(source_code))
    # 扁平化嵌套的AST节点
    s = checknesteddic(tree)
    root_node = tree['body'][0]
    # for i in nodelists:
    #     print(i)
    return nodelists
source1 = """def check_parentheses(s):
    stack = []
    for (id, i) in enumerate(s):
        if i in ['(', '[', '{']:
            stack.append(i)
        elif i in [')', ']', '}']:
            if not stack:
                return False
            elif stack[-1] == '(' and i == ')':
                stack.pop()
            elif stack[-1] == '[' and i == ']':
                stack.pop()
            elif stack[-1] == '{' and i == '}':
                stack.pop()
            else:
                return id
    return not stack"""


# source2 = """def writeBoolean(self, n):\n        \"\"\"\n        Writes a Boolean to the stream.\n        \"\"\"\n        t = TYPE_BOOL_TRUE\n\n        if n is False:\n            t = TYPE_BOOL_FALSE\n\n        self.stream.write(t)"""
# nodelists = parse_ast_python(source1)
# # parse_ast_python(source2)
# for i in nodelists:
#     print(i)