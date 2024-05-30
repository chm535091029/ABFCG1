import javalang
import json
from tqdm import tqdm
import collections
import sys
import astor

def get_name(obj):
    if(type(obj).__name__ in ['list', 'tuple']):
        a = []
        for i in obj:
            a.append(get_name(i))
        return a
    elif(type(obj).__name__ in ['dict', 'OrderedDict']):
        a = {}
        for k in obj:
            a[k] = get_name(obj[k])
        return a
    elif(type(obj).__name__ not in ['int', 'float', 'str', 'bool']):
        return type(obj).__name__
    else:
        return obj

def process_source(code):
    # with open(file_name, 'r', encoding='utf-8') as source:
    #     lines = source.readlines()
    #     print(lines)
    # with open(save_file, 'w+', encoding='utf-8') as save:
    code_list = code.split("\n")
    tks = []
    for line in code_list:
        code = line.strip()
        # print(code)
        # 使用了 javalang 库中的 tokenizer 对代码行进行分词处理，根据不同类型的 token 分别将其替换为相应的标识符
        tokens = list(javalang.tokenizer.tokenize(code))

        for tk in tokens:
            if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                tks.append('STR_')
            elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                tks.append('NUM_')
            elif tk.__class__.__name__ == 'Boolean':
                tks.append('BOOL_')
            else:
                tks.append(tk.value)
        # save.write(" ".join(tks) + '\n')
    return " ".join(tks)
def parse_ast_java(code):



    # with open(file_name, 'r', encoding='utf-8') as f:
    #     line = f.read()
    # with open(w, 'w+', encoding='utf-8') as wf:
    #  Leaf node（叶节点）的数量，这些 Leaf node 没有对应的值。
    ign_cnt = 0
    source_code = code.split("\n")
    code = process_source(code)
    code = code.strip()
    # print(code)
    tokens = javalang.tokenizer.tokenize(code)
    # print(tokens)
    token_list = list(javalang.tokenizer.tokenize(code))
    # print(token_list)
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    flatten = []
    try:
        tree = parser.parse_member_declaration()
        for path, node in tree:
            flatten.append({'path': path, 'node': node})
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        print("err")
        raise

        # print(code)
    ign = False
    outputs = []
    stop = False
    for i, Node in enumerate(flatten):

        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        # print(node)
        if node is not None and hasattr(node, 'position') and node.position is not None:
            source_fragment = source_code[node.position[0]-1:node.position[1]-1]
            # print(source_fragment)
            # print(i)
        # source_fragment = node.to_source().strip()
        children = []
        for child in node.children:
            child_path = None
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)

        d["id"] = i
        d["type"] = get_name(node)
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):

                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'
        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            # print('Leaf has no value!')
            # print(type(node))
            # print(code)
            ign = True
            ign_cnt += 1  #  Leaf node（叶节点）的数量，这些 Leaf node 没有对应的值。
            # break
        outputs.append(d)
    if not ign:

        # wf.write(json.dumps(outputs))
        return outputs
        # wf.write('\n')
    # print(ign_cnt)



# if __name__ == '__main__':
#     # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_
#     input_path = 'demo.java'
#     output_path = 'source.code'
#     ast_path = 'ast.txt'
#     process_source(input_path, output_path)
#     # generate ast file for source code
#     get_ast(output_path, ast_path)
# with open("deepcom-train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # list[dict{}]
# f.close()
# print(data[2]["code"])
# print(parse_ast_java(data[2]["code"]))