# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json


from py2neo import Graph, Node, Relationship, NodeMatcher

# from CloneDetect.get_clone_dict import get_clone_relation
from get_python_ast import parse_ast_python

import sys
from get_java_ast import *
# from pythonAST1.get_python_ast import *
from get_java_adg import *
from get_python_adg import *
import nltk
from get_nl_similarity import *
from get_python_cfg import *
from get_java_cfg import *
from get_java_pdg1 import *
def add_NL_PL_python_cosqa(address,auth,filename):
    graph = Graph(address,auth=auth)
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f) #list[dict{}]
    f.close()
    suc_count = 0
    fail_count = 0
    fall_method, behind_call_relations, forward_call_relations = parse_adg_python(filename)
    # 创建节点
    for dict in tqdm(data,desc="adding nodes and relations"):

        try:
            method_name=get_method_name_python(dict["code"])
            # ast_nodes = [json.loads(node.decode('utf-8')) if isinstance(node, bytes) else node for node in ast_nodes]
            # print(dict["idx"])
            ast_nodes = parse_ast_python(dict["code"].encode("utf-8"))
            cfgnodes,cfgedges = parse_cfg_python(dict["code"].encode("utf-8"))
            NL = Node("NL",nl=dict["doc"],docstring_tokens=dict["docstring_tokens"], train="True" if "train" in filename else "False",source=filename.split(".")[0],idx=dict["idx"].split("-")[-1])
            PL = Node("PL",code=dict["code"],code_tokens=dict["code_tokens"],code_ast=json.dumps(ast_nodes),code_cfg_blocks=json.dumps(cfgnodes),code_cfg_edges=json.dumps(cfgedges),
                      code_adg=json.dumps([{"fcall":forward_call_relations[method_name] if method_name in forward_call_relations.keys() else [],"bcall":behind_call_relations[method_name] if method_name in behind_call_relations.keys() else []}]),
                      train="True" if "train" in filename else "False",lang="python",source=filename.split(".")[0],idx=dict["idx"].split("-")[-1])

            # 添加节点到数据库中
            graph.create(NL)
            graph.create(PL)
            if dict["label"]==0:
                rel1 = Relationship(NL,"related",PL,label=dict["label"],idx=dict["idx"].split("-")[-1],train="True" if "train" in filename else "False")
                rel2 = Relationship(PL, "related", NL, label=dict["label"], idx=dict["idx"].split("-")[-1],
                                   train="True" if "train" in filename else "False")
            else:
                rel1 = Relationship(NL,"unrelated",PL,label=dict["label"],idx=dict["idx"].split("-")[-1],train="True" if "train" in filename else "False")
                rel2= Relationship(PL,"unrelated",NL,label=dict["label"],idx=dict["idx"].split("-")[-1],train="True" if "train" in filename else "False")

            graph.create(rel1)
            graph.create(rel2)
            suc_count+=1
        except Exception as e:
            # print(e)
            fail_count+=1
            pass

    print(str(suc_count)+"nl-pl pairs have been added while "+ str(fail_count) + " have been ignored")

def add_NL_PL_java_deepcom(address,auth,filename):
    graph = Graph(address, auth=auth)
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)  # list[dict{}]
    f.close()
    suc_count = 0
    fail_count = 0
    fall_method, behind_call_relations, forward_call_relations = parse_adg_java(filename)
    for i,dict in tqdm(enumerate(data),desc="adding nodes and relations"):
        try:
            tokens = nltk.word_tokenize(dict["code"])
            code_tokens = ' '.join(tokens)
            ast_dict = parse_ast_java(dict["code"])
            # method_name = get_method_name_java(dict["code"])
            # cfg_dicts = parse_cfg_java(dict["code"])
            # print(method_name)
            # pdg_nodes, pdg_edges = parse_pdg_java(dict["code"])
            NL = Node("NL",nl=dict["nl"],train="True" if "train" in filename else "False",source=filename.split(".")[0],idx=str(i))
            PL = Node("PL",code=dict["code"],train="True" if "train" in filename else "False",lang="java",code_tokens=code_tokens,code_ast=json.dumps(ast_dict),
                      # code_adg=json.dumps([{"fcall":forward_call_relations[method_name] if method_name in forward_call_relations.keys() else [],"bcall":behind_call_relations[method_name] if method_name in behind_call_relations.keys() else []}]),
                      # code_cfg=json.dumps(cfg_dicts),source=filename.split(".")[0],idx=str(i)
                      )
            rel1 = Relationship(NL, "related", PL,idx=str(i),train="True" if "train" in filename else "False")
            rel2 = Relationship(PL, "related", NL,idx=str(i),train="True" if "train" in filename else "False")
            graph.create(NL)
            graph.create(PL)
            graph.create(rel1)
            graph.create(rel2)

            suc_count+=1
        except Exception as e:
            print(e)
            fail_count+=1
            # pass
    print(str(suc_count)+"nl-pl pairs have been added while "+ str(fail_count) + " have been ignored")

def add_NL_PL_java_concode(address,auth,filename):
    graph = Graph(address, auth=auth)
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)  # list[dict{}]
    f.close()
    suc_count = 0
    fail_count = 0
    fall_method, behind_call_relations, forward_call_relations = parse_adg_java(filename)
    for i,dict in tqdm(enumerate(data),desc="adding nodes and relations"):
        try:
            # tokens = nltk.word_tokenize(dict["code"])
            # code_tokens = ' '.join(tokens)
            ast_dict = parse_ast_java(dict["code"])
            method_name = get_method_name_java(dict["code"])
            cfg_dicts = parse_cfg_java(dict["code"])
            if ast_dict == None or cfg_dicts==None:
                fail_count += 1
                continue
            # print(method_name)
            # pdg_nodes, pdg_edges = parse_pdg_java(dict["code"])
            NL = Node("NL",nl=dict["nl"].split("concode_field_sep")[0],train="True" if i<8000 else "False",source="concode",idx=str(i))
            PL = Node("PL",code=dict["code"],train="True" if i<8000 else "False",lang="java",code_tokens=dict["code"].split(),code_ast=json.dumps(ast_dict),
                      # code_adg=json.dumps([{"fcall":forward_call_relations[method_name] if method_name in forward_call_relations.keys() else [],"bcall":behind_call_relations[method_name] if method_name in behind_call_relations.keys() else []}]),
                      # code_cfg=json.dumps(cfg_dicts),source="concode",idx=str(i)
                      )
            rel1 = Relationship(NL, "related", PL,idx=str(i))
            rel2 = Relationship(PL, "related", NL,idx=str(i))
            graph.create(NL)
            graph.create(PL)
            graph.create(rel1)
            graph.create(rel2)
            suc_count+=1
        except Exception as e:
            # print(e)
            fail_count+=1
            # pass
    print(str(suc_count)+"nl-pl pairs have been added while "+ str(fail_count) + " have been ignored")

def add_pdg4java(address, auth,file_dir, dataset, train):
    graph = Graph(address, auth=auth)
    matcher = NodeMatcher(graph)
    pdg_output_list = os.listdir(file_dir)
    for dir in tqdm(pdg_output_list,desc="add pdg for java"):
        if dir.endswith('.bin') or dir.endswith('cpg'):
            continue
        dot = os.path.join(file_dir,dir,"0-pdg.dot")
        nodes, edges = read_pdg_result(dot)
        node = matcher.match("PL",idx=dir.split("_")[-1], source=dataset, train=train).first()
        if node is not None:
            node["pdg_nodes"] = json.dumps(nodes)
            node["pdg_edges"] = json.dumps(edges)
            graph.push(node)


# def add_clone_relations(address, auth, xmlname, dataset, train):
#     clone_dict = get_clone_relation(xmlname)
#     graph = Graph(address, auth=auth)
#     for class_id, num_lst in tqdm(clone_dict.items(), desc="adding clone relation"):
#         # print(k,v)
#         CLONE = Node("CLONE",clone_id=class_id,source=dataset,train=train)
#         for id in num_lst:
#             matcher = NodeMatcher(graph)
#             node = matcher.match("PL", idx=id, source=dataset, train=train).first()
#             if node is not None:
#                 rel = Relationship(node, "clone", CLONE, source=dataset, train=train)
#                 graph.create(rel)


def add_nl_similarity_relations(address,auth, filename,dataset):
    graph = Graph(address, auth=auth)
    matcher = NodeMatcher(graph)

    similar_nl_idx_groups = parse_nl_similarity(filename)

    for similar_idx,cluster in tqdm(enumerate(similar_nl_idx_groups),desc="adding similarity relations"):
        SIMILAR_TEXT = Node("SIMILAR_TEXT", source=dataset, similar_idx=str(similar_idx),similar_node=json.dumps(cluster),lang="java")
        for i in range(len(cluster)):
            similar_NL_node = matcher.match("NL",idx=str(cluster[i])).first()
            if similar_NL_node:
                rel = Relationship(similar_NL_node,"text_similar",SIMILAR_TEXT)
                graph.create(rel)


if __name__ == '__main__':
    nltk.data.path.append("punkt")

    add_NL_PL_java_deepcom("bolt://localhost:7687",auth=("neo4j", "password"),filename="deepcom.json")
    add_nl_similarity_relations("bolt://localhost:7687", auth=("neo4j", "password"), filename="deepcom.json",
                                dataset="codesearchnet")



