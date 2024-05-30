import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import json
from tqdm import tqdm
import io

import tokenize
import random
from py2neo import Graph, Node, Relationship, NodeMatcher
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torch
from Embedding4ast import get_ast_feature
from scipy.sparse import coo_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
code_tokens = []

#
class MyDataset(Dataset):
    def __init__(self, source_name,dataset_path_name,k=2,code_token_max=128):
        # Initialize dataset attributes and load necessary files
        self.dataset_path_name = dataset_path_name
        self.k = k
        self.code_token_max = code_token_max
        self.source_name = source_name

        with open(self.dataset_path_name, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        if "train" in dataset_path_name:
            guid_path = dataset_path_name.replace("ids", "guid").replace("json", "txt")
            self.suf = "train"
        elif "valid" in dataset_path_name:
            guid_path = dataset_path_name.replace("ids", "guid").replace("json", "txt").replace("valid", "nontrain")
            self.suf = "nontrain"
        elif "test" in dataset_path_name:
            guid_path = dataset_path_name.replace("ids", "guid").replace("json", "txt").replace("test", "nontrain")
            self.suf = "nontrain"
        with open(guid_path, "r") as fg:
            self.guids = {}
            for guid in fg:
                idx = guid.split("-")[0]
                i = guid.split("-")[1]
                if idx in self.guids:
                    self.guids[idx].append(i.strip())
                else:
                    self.guids[idx] = [i.strip()]

        # Load other necessary files or perform additional initialization if needed

    def __len__(self):
        # Return the size of the dataset
        return len(self.samples)

    def __getitem__(self, index):
        # Implement logic to load a single sample based on the given index

        sample = self.samples[index]
        nl_data = torch.tensor(sample["nl"], dtype=torch.long)
        codetoken_data = torch.tensor(sample["code_tokens"], dtype=torch.long)
        related_code_data = torch.tensor(sample["related_code"], dtype=torch.long)
        related_nl_data = torch.tensor(sample["related_nl"], dtype=torch.long)
        related_ast_data = torch.tensor(sample["related_ast"], dtype=torch.long)

        # Load or generate related_ast_matrice based on your logic
        ast_tokens_matrix = np.zeros([self.k, 2 * self.code_token_max, 2 * self.code_token_max]).astype(bool)
        for i, node in enumerate(sample["related_ast"]):
            if i < self.k:
                try:

                    np_file = f"{'/'.join(self.dataset_path_name.split('/')[:-1])}/{self.source_name}-{self.suf}_att/{sample['idx']}-{self.guids[str(sample['idx'])][i]}.npz"

                    # Load data using your _load_data function or other logic
                    loaded_data = np.load(np_file)
                    ast_tokens_matrix[i] = coo_matrix((loaded_data['data'], (loaded_data['row'], loaded_data['col'])),
                                                  shape=loaded_data['shape']).toarray()
                except:
                    # print(sample['idx'])
                    # print(self.guids[sample['idx']])
                    pass

        related_ast_matrice = torch.tensor(ast_tokens_matrix, dtype=torch.bool)

        return nl_data, codetoken_data, related_code_data, related_nl_data, related_ast_data, related_ast_matrice


def collate_fn(batch, max_nl_length=32, max_code_length=128, max_ast_length=256):
    # 获取每个样本的当前长度
    # nl_lengths = [torch.argmax((sample[0] == 0).to(torch.int32), dim=-1) for sample in batch]
    code_lengths = [torch.argmax((sample[1] == 0).to(torch.int32), dim=-1) for sample in batch]

    related_code_lengths = []
    related_nl_lengths = []
    related_ast_lengths = []

    for sample in batch:
        for i in range(sample[2].size(0)):
            related_code_lengths.append(torch.argmax((sample[2][i] == 0).to(torch.int32), dim=-1))
            related_nl_lengths.append(torch.argmax((sample[3][i] == 0).to(torch.int32), dim=-1))
            related_ast_lengths.append(torch.argmax((sample[4][i] == 0).to(torch.int32), dim=-1))


    for sample in batch:
        for i in range(sample[2].size(0)):
            if 0 in sample[2][i]:
                related_code_lengths.append(torch.argmax((sample[2][i] == 0).to(torch.int32), dim=-1))
            else:
                related_code_lengths.append(torch.tensor(max_code_length, dtype=torch.int32))
            if 0 in sample[3][i]:
                related_nl_lengths.append(torch.argmax((sample[3][i] == 0).to(torch.int32),dim=-1))
            else:
                related_nl_lengths.append(torch.tensor(max_code_length, dtype=torch.int32))
            if 0 in sample[4][i]:
                related_ast_lengths.append(torch.argmax((sample[4][i] == 0).to(torch.int32),dim=-1))
            else:
                related_ast_lengths.append(torch.tensor(max_code_length, dtype=torch.int32))
    # 计算需要截断的长度
    # max_nl_length = min(max(nl_lengths), max_nl_length)
    max_code_length = min(max(code_lengths), max_code_length)
    max_related_code_length = min(max(related_code_lengths), max_code_length)
    max_related_nl_length = min(max(related_nl_lengths), max_nl_length)
    max_related_ast_length = min(max(related_ast_lengths), max_ast_length)

    # 其他逻辑...

    # 对每个样本进行截断或填充
    padded_batch = [(
        sample[0][:],
        sample[1][:max_code_length],
        sample[2][:,:max_related_code_length],
        sample[3][:,:],
        sample[4][:,:max_related_ast_length],
        sample[5][:,:max_related_ast_length,:max_related_ast_length],
    ) for sample in batch]

    # 创建 padded_batch 的张量
    nl_data, codetoken_data, related_code_data, related_nl_data, related_ast_data, related_ast_matrice = zip(*padded_batch)
    nl_data = torch.stack(nl_data)
    codetoken_data = torch.stack(codetoken_data)
    related_code_data = torch.stack(related_code_data)
    related_nl_data = torch.stack(related_nl_data)
    related_ast_data = torch.stack(related_ast_data)
    related_ast_matrice = torch.stack(related_ast_matrice)

    return nl_data, codetoken_data, related_code_data, related_nl_data, related_ast_data, related_ast_matrice



class Batch4ABFCG():
    def __init__(self,dataset,source_name,address,auth):

        self.graph = Graph(address, auth=auth)
        self.matcher = NodeMatcher(self.graph)
        self.dataset = dataset
        # self.train = train
        self.source_name = source_name
        self.cnt_total = 0



    # def create_ast_matrix(self,ast_info, ast_node_max, ast_value_max, code_ast_vocab_stoi):
    #     if isinstance(code_ast_vocab_stoi, str):  # 判断code_tokens_vocab_stoi是否为文件名
    #         with open(code_ast_vocab_stoi, 'r') as file:
    #             code_ast_vocab_stoi = json.load(file)
    #     ast_matrix = np.zeros([ast_node_max, ast_node_max + ast_value_max + 1]).astype(int)
    #     # type_list = ['IfStatement', 'VariableDeclaration', 'ExplicitConstructorInvocation',
    #     #              'Annotation', 'SwitchStatementCase', 'VoidClassReference', 'BinaryOperation',
    #     #              'ForControl', 'ReturnStatement', 'MethodInvocation', 'SynchronizedStatement',
    #     #              'ReferenceType', 'CatchClauseParameter', 'Statement', 'SuperMemberReference', 'TryResource',
    #     #              'ContinueStatement', 'ArraySelector', 'ClassReference', 'Assignment', 'TypeArgument',
    #     #              'ClassCreator', 'ArrayCreator', 'ThrowStatement', 'WhileStatement', 'ElementValuePair',
    #     #              'ElementArrayValue', 'EnhancedForControl', 'StatementExpression', 'CatchClause', 'TryStatement',
    #     #              'BreakStatement', 'FormalParameter', 'BlockStatement', 'TernaryExpression', 'MemberReference',
    #     #              'BasicType', 'VariableDeclarator', 'AssertStatement', 'ConstructorDeclaration',
    #     #              'LocalVariableDeclaration', 'ForStatement', 'SuperMethodInvocation', 'MethodDeclaration',
    #     #              'ArrayInitializer', 'Literal', 'TypeParameter', 'SuperConstructorInvocation', 'DoStatement',
    #     #              'InnerClassCreator', 'SwitchStatement', 'Cast', 'This','LambdaExpression','MethodReference','InferredFormalParameter',
    #     #              'FieldDeclaration','ClassDeclaration']
    #     type_list = ['Module', 'Interactive', 'Expression', 'Suite', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef',
    #                  'Return', 'Delete', 'Assign', 'AugAssign', 'AnnAssign', 'For', 'AsyncFor', 'While', 'If',
    #                  'With', 'AsyncWith', 'Raise', 'Try', 'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal',
    #                  'Expr', 'Pass', 'Break', 'Continue', 'BoolOp', 'NamedExpr', 'BinOp', 'UnaryOp', 'Lambda',
    #                  'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp', 'Await', 'Yield',
    #                  'YieldFrom', 'Compare', 'Call', 'FormattedValue', 'JoinedStr', 'Constant', 'Attribute',
    #                  'Subscript', 'Starred', 'Name', 'List', 'Tuple','arguments','arg','alias']
    #
    #     type_dict = {v: type_ids + 1 for type_ids, v in enumerate(type_list)}
    #     for node in ast_info:
    #         if int(node["id"]) < ast_node_max:
    #             try:
    #                 for child in node["children"]:
    #                     if child < ast_node_max:
    #                         ast_matrix[node["id"]][child] = 1
    #             except:
    #                 pass
    #             try:
    #                 ast_matrix[int(node["id"])][ast_node_max] = code_ast_vocab_stoi["<bos>"]
    #                 value_tokens_len = len(word_tokenize(node["value"]))
    #                 value_tokens = word_tokenize(node["value"])[:min(value_tokens_len, ast_value_max-2)]
    #                 ast_matrix[int(node["id"])][ast_node_max + 1:ast_node_max + 1 + value_tokens_len] = [code_ast_vocab_stoi[value_token] if value_token in code_ast_vocab_stoi.keys() else code_ast_vocab_stoi["<unk>"] for value_token in value_tokens]
    #                 ast_matrix[int(node["id"])][ast_node_max + 1 + min(len(word_tokenize(node["value"])), ast_value_max)] = code_ast_vocab_stoi["<eos>"]
    #                 # ast_matrix[node["id"]][ast_node_max + 1 + min(len(word_tokenize(node["value"])),ast_value_max) + 1:ast_node_max + ast_value_max] = code_ast_vocab_stoi["<pad>"]
    #             except:
    #                 ast_matrix[int(node["id"])][ast_node_max] = code_ast_vocab_stoi["<bos>"]
    #                 # ast_matrix[node["id"]][ast_node_max + 1:ast_node_max + ast_value_max - 2] = code_ast_vocab_stoi["<pad>"]
    #                 ast_matrix[int(node["id"])][ast_node_max + 1:ast_node_max + ast_value_max - 1] = code_ast_vocab_stoi["<eos>"]
    #             try:
    #                 ast_matrix[int(node["id"])][ast_node_max + ast_value_max] = type_dict[node["type"]]
    #             except:
    #                 # print(type(node['type']))
    #                 try:
    #                     type_dict[node['type']] = len(type_dict)+1
    #                     ast_matrix[int(node["id"])][ast_node_max + ast_value_max] = type_dict[node["type"]]
    #                 except:
    #                     return None
    #     return ast_matrix

    def create_ast_matrix(self,ast):
        # 初始化用于存储数据流关系的图
        data_flow_graph = {}
        # subword_map = {}  # word->subword
        type_map = {}  # type->word
        node_map = {}  # node->type
        type_id = 0
        word_id = 0

        # 遍历AST节点
        ast_token = []
        # 元素为1的位置是AST的边，元素为2的位置是控制流边，元素为3的位置是数据流的边，控制流和数据流都包含了AST的边
        att_matrix = np.zeros((256, 256), dtype=bool)

        next_siblings = {}  # node_id
        leave_id = []
        edges = set()
        # data_flow = set()
        # control_flow = set()
        for node in ast:

            type_id = len(ast_token)
            node_map[str(node['id'])] = type_id
            ast_token.append(node["type"])

            # type_map[str(type_id)] = []

            if "value" in node.keys():
                word_id = len(ast_token)
                type_map[str(type_id)] = word_id
                ast_token.append(node["value"])
                # subword_map[str(word_id)] = []
                # if len(wordninja.split(node['value'])) > 1:
                #     for subword in wordninja.split(node['value']):
                #         subword_map[str(word_id)].append(len(ast_token))
                #         ast_token.append(subword)
            # elif "value" not in node.keys:
            #
            if 'children' in node.keys():
                for i, child in enumerate(node['children'][:-1]):
                    next_siblings[str(child)] = node['children'][i + 1]

        for node in ast:
            if "children" in node.keys():
                for child in node["children"]:
                    edges.add((node_map[str(node['id'])], node_map[str(child)]))
                    # edges.add((node_map[str(child)],node_map[str(node['id'])]))
                    if "children" not in ast[child].keys():
                        try:
                            edges.add((type_map[str(node_map[str(node['id'])])], type_map[str(node_map[str(child)])]))
                            # edges.add((type_map[str(node_map[str(child)])],type_map[str(node_map[str(node['id'])])]))
                        except:
                            pass
                    #     print(child)
                    #     print(node['id'])
                    #     print(node_map)
                    #     print(type_map)
                    #     print(ast_token)
                    #     raise
            else:
                leave_id.append(type_map[str(node_map[str(node['id'])])])
        # 加一条自己到自己的边
        for i, token in enumerate(ast_token):
            edges.add((i, i))
        # 给叶子节点加边
        for i in range(len(leave_id) - 1):
            edges.add((leave_id[i], leave_id[i + 1]))
            # edges.add((leave_id[i+1],leave_id[i]))

        # add edges between value token and its subtokens
        # for key, values in subword_map.items():
        #     for value in values:
        #         edges.add((int(key), value))
                # edges.add((value,int(key)))
                # data_flow.add((int(key),value))
        # add edges between type token and its value
        for key, value in type_map.items():
            edges.add((int(key), value))
            # edges.add((value,int(key)))

        for node in ast:
            last_child = None
            if "children" in node.keys():
                for child in node["children"]:
                    # edges.add((node_map[str(node['id'])],node_map[str(child)]))
                    if node["type"] == "ForStatement" and last_child is not None:
                        edges.add((node_map[str(last_child)], node_map[str(child)]))
                        # edges.add((node_map[str(child)],node_map[str(last_child)]))
                        try:
                            edges.add((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])]))
                            # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                        except:
                            pass
                    elif node["type"] == "BlockStatement" and last_child is not None:
                        edges.add((node_map[str(last_child)], node_map[str(child)]))
                        # edges.add((node_map[str(child)],node_map[str(last_child)]))
                    elif node["type"] == "WhileStatement" and last_child is not None:
                        edges.add((node_map[str(last_child)], node_map[str(child)]))
                        # edges.add((node_map[str(child)],node_map[str(last_child)]))
                        try:
                            edges.add((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])]))
                            # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                        except:
                            pass
                    elif node["type"] == "IfStatement" and last_child is not None:
                        edges.add((node_map[str(last_child)], node_map[str(child)]))
                        # edges.add((node_map[str(child)],node_map[str(last_child)]))
                        try:
                            edges.add((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])]))
                            # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                        except:
                            pass

                    last_child = child
                    # try:
                    if node['type'] == 'SwitchStatement' and ast[child]['type'] == 'SwitchStatementCase' and str(
                            node['id']) in next_siblings and 'children' in ast[child]:
                        for i in ast[child]['children']:
                            if 'BreakStatement' in ast[i]['type']:
                                edges.add((node_map[str(i)], node_map[str(next_siblings[str(node['id'])])]))
                                # edges.add((node_map[str(next_siblings[str(node['id'])])],node_map[str(i)]))
                    # except:
                    #     print('node_map:'+str(node_map))
                    #     print('node_map[str(i)]:'+str(node_map[str(i)]))
                    #     print('next_siblings:'+str(next_siblings))
                    #     print("node['id']:"+str(node['id']))
                    #     print("i:"+str(i))
                    #     raise
        last_use = {}
        for node in ast:
            # 如果是变量定义节点，将其添加到图中
            if node["type"] == "FormalParameter":
                variable_name = node["value"]
                data_flow_graph[variable_name] = type_map[str(node_map[str(node["id"])])]

            # 如果是变量使用节点，将其添加到对应变量的usages中
            elif node["type"] in ["ReferenceType", "MemberReference", "MethodInvocation"]:
                variable_name = node["value"]
                if variable_name in data_flow_graph:
                    edge1 = (data_flow_graph[variable_name], type_map[str(node_map[str(node["id"])])])
                    # edge2 = (type_map[str(node_map[str(node["id"])])],data_flow_graph[variable_name])
                    edges.add(edge1)
                    # edges.add(edge2)
                    last_use[variable_name] = type_map[str(node_map[str(node["id"])])]
                if variable_name in last_use:
                    edge1 = (last_use[variable_name], type_map[str(node_map[str(node["id"])])])
                    # edge2 = (type_map[str(node_map[str(node["id"])])],last_use[variable_name])
                    edges.add(edge1)
                    # edges.add(edge2)
        # for i, edge_set in enumerate([edges,control_flow,data_flow]):
        #     for edge in edge_set:
        #         try:
        #             att_matrix[i,edge[0],edge[1]] = True
        #         except:
        #             pass
        for edge in edges:
            try:
                att_matrix[edge[0], edge[1]] = True
            except:
                pass
        # for edge in data_flow:
        #     try:
        #         if att_matrix[edge[0], edge[1]] != 1:
        #             att_matrix[edge[0], edge[1]] = 3
        #     except:
        #         pass
        # for edge in control_flow:
        #     try:
        #         if att_matrix[edge[0], edge[1]] != 1:
        #             att_matrix[edge[0], edge[1]] = 2
        #     except:
        #         pass
        return ast_token, coo_matrix(att_matrix)

    def get_text_related_code_tokens(self,k,samples,train_samples=None):
        # for skip in tqdm(range(0,self.cnt_total, 1000), desc="collecting code with similar texts batch by batch"):
        train = False
        if train_samples is None:
            train = True
            train_samples = samples
            # code_asts = open("code_asts.txt", "a")
            nl = open("nl_text.txt", "a")


        skip = 0


        global code_tokens

        while True:
            query = f"""
                        MATCH (s:SIMILAR_TEXT)
                        WHERE s.source = '{self.source_name}'
                        RETURN s
                        ORDER BY toInteger(s.similar_idx)
                        SKIP {skip}
                        LIMIT 500
                    """
            # query2 = f"""
            #             MATCH (n:NL)-[r:text_similar]-(s:SIMILAR_TEXT)-[d:text_similar]-(m:NL)-[u:related]-(o:PL)
            #             WHERE n.source = '{self.source_name}' AND n.train = '{self.train}' AND n.idx = '{idx}' AND m.train ='True'
            #             RETURN o,m
            #             LIMIT {k}
            #             """
            # 执行查询
            result = self.graph.run(query)
            skip+=500
            # has_related = True
            if result.forward()==False:
                break
            for node in tqdm(result,"finding related nl and code batch by batch ..."):
                node_list = json.loads(node[0]["similar_node"])
                for id1 in node_list:
                    if str(id1) in samples.keys():
                        similar_nl = []
                        similar_code = []
                        similar_ast = []
                        # similar_cfg = []
                        i = 0
                        for id2 in node_list:
                            if id1!=id2 and i<k and str(id2) in train_samples.keys():
                                similar_nl.append(train_samples[str(id2)]["nl"])
                                similar_code.append(train_samples[str(id2)]["code_tokens"])
                                similar_ast.append(train_samples[str(id2)]["code_ast"])
                                # similar_cfg.append(train_samples[str(id2)]["code_cfg"])
                                i+=1
                        if len(similar_nl) and len(similar_code):
                            samples[str(id1)]["related_nl"] = similar_nl
                            samples[str(id1)]["related_code"] = similar_code
                            samples[str(id1)]["related_ast"] = similar_ast
                            # samples[str(id1)]["related_cfg"] = similar_cfg

                            # code_tokens.write(' '.join(sample["code_tokens"])+' ')
                            if train is True:
                                nl.write(samples[str(id1)]["nl"] + " ")
                                for token in samples[str(id1)]["code_tokens"]:
                                    code_tokens.append(token)
                                # shared_nl_code.write(sample["nl"] + " " + ' '.join(sample["code_tokens"]) + ' ')
                                for ast_node in samples[str(id1)]["code_ast"]:
                                    try:
                                        if "type" in ast_node:
                                            code_tokens.append(ast_node["type"])
                                        if "value" in ast_node:
                                            code_tokens.append(ast_node["value"])
                                    except:
                                        pass
        # shared_nl_code.close()
        # code_cfgs.close()
        return samples
    def python_tokenizer(self,code):
        code = code.split("\n")
        flag = 0

        rm_list = []
        for i, c in enumerate(code):

            if ('\"\"\"' in c or '\'\'\'' in c) and flag ==0:
                flag += 1
            elif ('\"\"\"' in c or '\'\'\'' in c)and flag >0:
                rm_list.append(i)
                flag -= 1
            if flag>0:
                rm_list.append(i)
            elif "#" in c:
                rm_list.append(i)
        for i, r in enumerate(rm_list):
            code.pop(r - i)
        # print(code)
        code = '\n'.join(code)
        code_tokens = []
        tokens = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
        for token in tokens:
            token_type, token_string, start, end, line = token
            code_tokens.append(token_string)
        code_tokens.pop(0)
        return code,code_tokens
    def search4samples_with_text_similar_relation(self):
        samples = {}
        skip = 0
        while True:
            query = f"""
                               MATCH (p:PL)-[r:related]->(n:NL)-[t:text_similar]-(s:SIMILAR_TEXT)
                               WHERE n.source = '{self.source_name}'
                               RETURN n,p
                               ORDER BY toInteger(n.idx)
                               SKIP {skip}
                               LIMIT 500
                               """
            # 执行查询
            result = self.graph.run(query)
            if result.forward() == False:
                break
            # 处理查询结果
            for node in result:

                try:
                    sample = {}
                    # pl_node = self.matcher.match("PL", idx=node["idx"],source=self.source_name, train=self.train).first()
                    # nl.write(" "+node[0]["nl"])
                    sample["nl"] = node[0]["nl"]

                    # code_tokens.write(" " + str(node[1]["code"]))
                    # try:
                    #     for ast_node in json.loads(node[1]["code_ast"]):
                    #         try:
                    #             code_asts.write(" " + ast_node["value"])
                    #         except:
                    #             pass
                    # except:
                    #     print(node[1]["idx"])
                    # smples[pl_node[0]["idx"]]["code"] = pl_node[0]["code"]
                    sample["code"], sample["code_tokens"] = self.python_tokenizer(node[1]["code"])
                    sample["code_ast"] = json.loads(node[1]["code_ast"])
                    # sample["code_cfg"] = json.loads(node[1]["code_cfg"])
                    # samples[node[0]["idx"]]["code_tokens"] = node[1]["code_tokens"]
                    sample["idx"] = node[1]["idx"]
                    # samples[node[0]["idx"]]["code"] = node[1]["code"]

                    # samples[node[0]["idx"]]["related_nl"] = [node[2]["nl"]]
                    # samples[node[0]["idx"]]["related_code"] = [node[3]["code"]]
                    samples[node[0]["idx"]] = sample
                except:
                    pass

            skip += 500
            if skip % 10000 == 0:
                print(f"now we are searching for {skip} samples")

        self.cnt_total = len(samples)
        print(f"got {self.cnt_total} samples")
        f = open(self.dataset.split(".")[0]+"_all_samples.json", "w")
        for s in samples.values():
            f.write(json.dumps(s))
            f.write("\n")
        f.close()
    def create_vocab_and_ids_of_dataset(self,nl_max,code_token_max,ast_node_max,ast_edge_max,ast_value_max,k,subset,
                                        train_dataset_name=None):
        """
            create the vocab for nl,pl,tokens,asts,and dump them to json files
        """

        samples = {}

        f = open(self.dataset.split(".")[0] + "_" + subset + "_samples.json", "r")
        sample = f.readline()

        while len(sample):
            sample = json.loads(sample)
            samples[sample["idx"]] = sample
            sample = f.readline()
        f.close()
        if train_dataset_name is not None:
            train_file = open(train_dataset_name,"r")
            train_samples = {}
            d = train_file.readline()
            while len(d):
                s = json.loads(d)
                train_samples[s["idx"]] = s
                d = train_file.readline()
        else:
            train_samples = None
        samples = self.get_text_related_code_tokens(k,samples,train_samples)
        print(len(code_tokens))
        f = open(self.dataset.split(".")[0]+"-"+subset+"_samples_new.json","w")
        for s in samples.values():
            f.write(json.dumps(s))
            f.write("\n")
        self.cnt_total = len(samples)
        print(self.cnt_total)

        del samples

        if train_dataset_name is None:
            nl = open("nl_text.txt","r")
            nl_texts = nl.read()
            tokens = word_tokenize(nl_texts)
            word_freq = FreqDist(tokens)

            filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= 3]

            sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
            nl_vocab_stoi = {}
            nl_vocab_itos = {}
            for id, (word, freq) in tqdm(enumerate(sorted_words), desc="creating vocab for nl"):
                nl_vocab_stoi[word] = id+3
                nl_vocab_itos[str(id+3)] = word
            nl_vocab_stoi["<unk>"] = len(sorted_words)+3
            nl_vocab_itos[str(len(sorted_words)+3)] = "<unk>"
            nl_vocab_stoi["<pad>"] = 0
            nl_vocab_itos[str(0)] = "<pad>"
            nl_vocab_stoi["<bos>"] = 1
            nl_vocab_itos[str(1)] = "<bos>"
            nl_vocab_stoi["<eos>"] = 2
            nl_vocab_itos[str(2)] = "<eos>"
            with open(self.dataset.split(".")[0]+"_nl_vocab.json","w") as f:
                json.dump({"nl_vocab_stoi":nl_vocab_stoi, "nl_vocab_itos":nl_vocab_itos},f)
            f.close()
        #

            tokens = code_tokens
            word_freq = FreqDist(tokens)

            filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= 3]

            sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
            code_tokens_vocab_stoi = {}
            code_tokens_vocab_itos = {}
            for id, (word, freq) in tqdm(enumerate(sorted_words), desc="creating vocab for code tokens"):
                code_tokens_vocab_stoi[word] = id+3
                code_tokens_vocab_itos[str(id+3)] = word
            code_tokens_vocab_stoi["<unk>"] = len(sorted_words) + 3
            code_tokens_vocab_itos[str(len(sorted_words) + 3)] = "<unk>"
            code_tokens_vocab_stoi["<pad>"] = 0
            code_tokens_vocab_itos[str(0)] = "<pad>"
            code_tokens_vocab_stoi["<bos>"] = 1
            code_tokens_vocab_itos[str(1)] = "<bos>"
            code_tokens_vocab_stoi["<eos>"] = 2
            code_tokens_vocab_itos[str(2)] = "<eos>"
            with open(self.dataset.split(".")[0]+"_code_tokens_vocab.json","w") as f:
                json.dump({"code_tokens_vocab_stoi":code_tokens_vocab_stoi,"code_tokens_vocab_itos":code_tokens_vocab_itos},f)
            f.close()


            # code_asts = open("code_asts.txt", "r")
            # asts_text = code_asts.read()
            # tokens = word_tokenize(asts_text)
            # word_freq = FreqDist(tokens)
            #
            # filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= 4]
            #
            # sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
            # code_ast_vocab_stoi = {}
            # code_ast_vocab_itos = {}
            # for id, (word, freq) in tqdm(enumerate(sorted_words), desc="creating vocab for asts"):
            #     code_ast_vocab_stoi[word] = int(id) + 3
            #     code_ast_vocab_itos[str(int(id)+3)] = word
            # code_ast_vocab_stoi["<unk>"] = len(sorted_words) + 3
            # code_ast_vocab_itos[str(len(sorted_words)+3)] = "<unk>"
            # code_ast_vocab_stoi["<pad>"] = 0
            # code_ast_vocab_itos[str(0)] = "<pad>"
            # code_ast_vocab_stoi["<bos>"] = 1
            # code_ast_vocab_itos[str(1)] = "<bos>"
            # code_ast_vocab_stoi["<eos>"] = 2
            # code_ast_vocab_itos[str(2)] = "<eos>"
            # with open(self.dataset.split(".")[0]+"_code_asts_vocab.json","w") as f:
            #     json.dump({"code_ast_vocab_stoi": code_ast_vocab_stoi, "code_ast_vocab_itos": code_ast_vocab_itos},f)
            # f.close()
            # nl.close()
            # code_asts.close()
        # os.remove("nl_text.txt")
        # os.remove("code_asts.txt")


        if train_dataset_name is not None:
            with open(self.dataset.split(".")[0]+"_nl_vocab.json", "r") as f:
                nl_vocab_stoi = json.load(f)["nl_vocab_stoi"]
            f.close()
            # with open(self.dataset.split(".")[0]+"_code_asts_vocab.json", "r") as f:
            #     code_ast_vocab_stoi = json.load(f)["code_ast_vocab_stoi"]
            # f.close()
            with open(self.dataset.split(".")[0]+"_code_tokens_vocab.json", "r") as f:
                code_tokens_vocab_stoi = json.load(f)["code_tokens_vocab_stoi"]
            f.close()
            # with open(train_dataset_name.split(".")[0].split("_")[0] + "_code_nl_tokens_vocab.json", "r") as f:
            #     code_nl_tokens_vocab_stoi = json.load(f)["code_nl_tokens_vocab_stoi"]
            # f.close()
        fp = open(self.dataset.split(".")[0] + "-" + subset+ "_samples_ids.json", "w")
        fg = open(self.dataset.split(".")[0] + "-" + subset+ "_samples_guid.txt","w")
        ft = open(self.dataset.split(".")[0] + "-" + subset+ "_samples.json","w")

        with open(self.dataset.split(".")[0]+ "-" + subset + "_samples_new.json","r") as f:

            for cnt in tqdm(range(0,self.cnt_total),desc="translating NL and IR to ids..."):
                sample = json.loads(f.readline())

                # nl_matrix,code_matrix,has_related = self.get_text_related_code_tokens(sample["idx"],k,code_tokens_vocab_stoi,nl_vocab_stoi,code_token_max,nl_max)
                # if has_related is False:
                #     continue
                # sample["related_nl"] = nl_matrix.tolist()
                # sample["related_code"] = code_matrix.tolist()
                if "related_nl" not in sample.keys() or "related_code" not in sample.keys():
                    continue
                ex = {}
                sample["idx"] = cnt
                tokens = word_tokenize(str(sample["nl"]))
                ids = []
                # ids.append(nl_vocab_stoi["<bos>"])
                for token in tokens[0:nl_max]:
                    ids.append(nl_vocab_stoi[token] if token in nl_vocab_stoi.keys() else nl_vocab_stoi["<unk>"])
                # ids.append(nl_vocab_stoi["<eos>"])
                ids.extend([nl_vocab_stoi["<pad>"] for i in range(nl_max-len(ids))])
                ex["nl"] = sample["nl"]
                sample["nl"] = ids

                tokens = sample["code_tokens"]

                ids = []
                ids.append(code_tokens_vocab_stoi["<bos>"])
                for token in tokens[0:code_token_max-2]:
                    ids.append(code_tokens_vocab_stoi[token] if token in code_tokens_vocab_stoi.keys() else code_tokens_vocab_stoi["<unk>"])
                ids.append(code_tokens_vocab_stoi["<eos>"])
                ids.extend([code_tokens_vocab_stoi["<pad>"] for i in range(code_token_max-len(ids))])
                ex["code_tokens"] = tokens
                sample["code_tokens"] = ids


                # ids = self.create_ast_matrix(sample["code_ast"],ast_node_max,ast_value_max,code_ast_vocab_stoi)
                # sample["code_ast"] = ids.tolist()
                # del sample["code_cfg"]
                # del sample["code_ast"]
                asts = []
                # ast_edges = []
                is_ast_available = True
                for num, ast in enumerate(sample["related_ast"]):
                    if num<k:
                        # ast = {}
                        ast_tokens,ast_matrix = self.create_ast_matrix(ast)
                        np.savez_compressed(f"{self.source_name}-{subset}_att/{cnt}-{num}.npz", data=ast_matrix.data, row=ast_matrix.row,
                                            col=ast_matrix.col,
                                            shape=ast_matrix.shape)
                        fg.write(f"{cnt}-{num}"+'\n')
                        if ast_matrix is None:
                            is_ast_available = False
                            break
                        asts.append(ast_tokens)
                        # ast_node_feature, ast_edge_feature = get_ast_feature(ast_matrix,ast_node_max,ast_edge_max)
                        # ast_node = ast_node_feature.tolist()
                        # ast_edge = ast_edge_feature.tolist()
                        # ids.append(ast_matrix.tolist())
                        # ast_nodes.append(ast_node)
                        # ast_edges.append(ast_edge)
                if is_ast_available is False:
                    continue
                # if len(asts)<k:
                #     for _ in range(k-len(asts)):
                #         asts.append(np.zeros_like(ast_tokens).tolist())
                        # ast_edges.append(np.zeros_like(ast_edge_feature).tolist())
                # sample["ast_node_feature"] = ast_nodes
                # sample["ast_edge_feature"] = ast_edges

                sample["related_ast"] = asts

                # del sample["related_cfg"]


                code_matrix = np.zeros([k, code_token_max]).astype(int)
                nl_matrix = np.zeros([k, nl_max]).astype(int)
                ast_tokens_matrix = np.zeros([k, 2*code_token_max]).astype(int)
                for i, node in enumerate(sample["related_nl"]):
                    if i < k:
                        nl_len = len(node)
                        word_list = node[:min(nl_len, nl_max)]
                        # nl_matrix[i][0] = nl_vocab_stoi["<bos>"]
                        nl_matrix[i][:len(word_list)] = [
                            nl_vocab_stoi[token] if token in nl_vocab_stoi.keys() else nl_vocab_stoi["<unk>"] for token
                            in word_list]
                        # nl_matrix[i][len(word_list)] = nl_vocab_stoi["<eos>"]

                for i, node in enumerate(sample["related_code"]):
                    if i < k:
                        code_len = len(node)
                        word_list = node[:min(code_len, code_token_max)]
                        # code_matrix[i][0] = code_tokens_vocab_stoi["<bos>"]
                        code_matrix[i][:len(word_list)] = [
                            code_tokens_vocab_stoi[token] if token in code_tokens_vocab_stoi.keys() else
                            code_tokens_vocab_stoi["<unk>"] for token in word_list]
                        # code_matrix[i][len(word_list)] = code_tokens_vocab_stoi["<eos>"]

                for i, node in enumerate(sample["related_ast"]):
                    if i < k:
                        code_len = len(node)
                        word_list = node[:min(code_len, 2*code_token_max)]
                        # ast_tokens_matrix[i][0] = code_tokens_vocab_stoi["<bos>"]
                        ast_tokens_matrix[i][:len(word_list)] = [
                            code_tokens_vocab_stoi[token] if token in code_tokens_vocab_stoi.keys() else
                            code_tokens_vocab_stoi["<unk>"] for token in word_list]
                        # ast_tokens_matrix[i][len(word_list)] = code_tokens_vocab_stoi["<eos>"]
                if np.all(nl_matrix==0) or np.all(code_matrix==0) or np.all(ast_tokens_matrix==0):
                    continue
                sample["related_nl"] = nl_matrix.tolist()
                sample["related_code"] = code_matrix.tolist()
                sample["related_ast"] = ast_tokens_matrix.tolist()

                fp.write(json.dumps(sample))
                fp.write("\n")

                ft.write(json.dumps(ex)+'\n')

        f.close()
        fp.close()

    def create_dataset(self,nl_max,code_token_max,ast_node_max,ast_edge_max,ast_value_max,k):

        self.search4samples_with_text_similar_relation()
        f = open(self.dataset.split(".")[0] + "_all_samples.json", "r")
        f1 = open(self.dataset.split(".")[0]+"_train_samples.json","w")
        f2 = open(self.dataset.split(".")[0]+"_nontrain_samples.json","w")
        all_samples = f.readlines()
        random.shuffle(all_samples)
        for i, sample in enumerate(all_samples):
            if i<len(all_samples)*0.8:
                f1.write(sample)
            else:
                f2.write(sample)
        f.close()
        f1.close()
        f2.close()
        self.create_vocab_and_ids_of_dataset(nl_max,code_token_max,ast_node_max,ast_edge_max,ast_value_max,k,"train")
        self.create_vocab_and_ids_of_dataset(nl_max,code_token_max,ast_node_max,ast_edge_max,ast_value_max,k,"nontrain",self.dataset.split(".")[0]+"-train_samples_new.json")
        f = open(self.dataset.split(".")[0] + "-nontrain_samples_ids.json","r")
        f_ = open(self.dataset.split(".")[0] + "-nontrain_samples.json","r")
        f_t = open(self.dataset.split(".")[0] + "-test_samples.json","w")
        f_v = open(self.dataset.split(".")[0] + "-valid_samples.json","w")
        f1 = open(self.dataset.split(".")[0] + "-valid_samples_ids.json","w")
        f2 = open(self.dataset.split(".")[0] + "-test_samples_ids.json","w")
        # all_samples = f.readlines()
        data1 = f.readlines()
        data2 = f_.readlines()
        permutation_indices = np.random.permutation(len(data2))
        all_samples1 = [data1[i] for i in permutation_indices]
        all_samples2 = [data2[i] for i in permutation_indices]
        for i, sample in enumerate(all_samples1):
            if i<len(all_samples1)*0.5:
                f1.write(sample)
            else:
                f2.write(sample)
        f.close()
        f1.close()
        f2.close()

        # all_samples = f_.readlines()
        for i, sample in enumerate(all_samples2):
            if i<len(all_samples2)*0.5:
                f_v.write(sample)
            else:
                f_t.write(sample)
        f_.close()
        f_v.close()
        f_t.close()
        os.remove(self.dataset.split(".")[0] + "_all_samples.json")
        os.remove(self.dataset.split(".")[0] + "-nontrain_samples_ids.json")
        os.remove(self.dataset.split(".")[0] + "-nontrain_samples.json")
        os.remove(self.dataset.split(".")[0]+"-train_samples_new.json")
        os.remove(self.dataset.split(".")[0]+"-nontrain_samples_new.json")
    def get_dataloader(self,dataset_path_name,batch_size,shuffle,k=2,code_token_max=128):
        dataset = MyDataset(self.source_name,dataset_path_name,k=2,code_token_max=128)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=shuffle, drop_last=False,num_workers=2,pin_memory=True)
        return dataloader


# 改一下存npz文件的路径
# batch = Batch4ABFCG("csn-java.json","codesearchnet","bolt://localhost:7687",auth=("neo4j", "password"))
# batch.create_dataset(32,128,92,32,4,2)


# batch.search4samples_with_text_similar_relation()
# batch.create_vocab_and_ids_of_dataset(32,128,92,32,4,2,"train")
# batch.create_vocab_and_ids_of_dataset(32,128,92,32,4,2,"nontrain","concode-train_samples_new.json")



