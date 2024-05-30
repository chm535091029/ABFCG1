import re
from tqdm import tqdm
import json


def file_dev(dataset):  #将代码中的换行符换成$
    program = []
    with open(dataset,"r",encoding="utf-8") as f:
        data = json.load(f)
        for dict in data:

            program.append(dict["code"].replace('\n', '$'))

    return program




# 函数首先使用正则表达式找到程序中所有函数的名称，并将其存储到func_name列表中。然后，函数在程序中查找所有调用其他函数的语句，
# 并将这些调用关系添加到call_relations列表中。
def get_calling_relationship(program):
    func_name = []  #统计所有的函数名
    behind_call_relations = {}  #统计所有的后向调用关系{k:[]}
    forward_call_relations = {}  # 统计所有的前向调用关系{k:[]}
    for code in tqdm(program,desc="get_calling_relationship"):
        code_lines = code.split("$")
        Fname = ""
        for i,code_line in enumerate(code_lines):
            if i == 0:#在代码的第一行
                function = re.findall(r"(\w+)\s*\(", code_line)
                Fname = function[0]
                if len(list(filter(None, Fname.split(' ')))) > 1 or Fname.replace(' ', '').isdigit() or 'if' == Fname.replace(' ', '') or 'for' == Fname.replace(' ', '') or 'while' == Fname.replace(' ',''):
                    continue
                else:
                    func_name.append(Fname.replace(" ","")) #解析到的第一个元素是函数名，其余的是参数中的API
                    if Fname not in behind_call_relations.keys():
                        behind_call_relations[Fname] = []

                    if len(function)>1: #将剩余的API加入到字典中
                        behind_call_relations[Fname].extend(function[1:-1])

                        #为剩余的API添加前向的调用字典键值对
                        for fcall in function[1:-1]:
                            func_name.append(fcall.replace(" ",""))
                            if fcall not in forward_call_relations.keys():
                                forward_call_relations[fcall] = [Fname]
                            else:
                                forward_call_relations[fcall].append(Fname)
            else:
                apis = re.findall(r"(\w+)\s*\(", code_line)
                # if code_line==">>> sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()":
                # print(apis)

                # print(code_line)
                for api in apis:
                    # print(api)
                    if len(list(filter(None, api.split(' ')))) > 1 or api.replace(' ','').isdigit() or 'if' == api.replace(' ', '') or 'for' == api.replace(' ', '') or 'while' == api.replace(' ', ''):
                        continue
                    else:
                        behind_call_relations[Fname].append(api)
                        func_name.append(api.replace(" ",""))
                        if api not in forward_call_relations.keys():
                            forward_call_relations[api] = [Fname]
                        else:
                            forward_call_relations[api].append(Fname)


    return list(filter(None, list(set(func_name)))), behind_call_relations,forward_call_relations


def parse_adg_python(filename):
    code = file_dev(filename)
    all_method, behind_call_relations,forward_call_relations = get_calling_relationship(code)
    # node_list, front_call, behind_call = build_instant_neighbour(call_relations)
    # front_call_dict, behind_call_dict = build_instant_neighbour_dict(node_list, front_call, behind_call)
    return all_method, behind_call_relations,forward_call_relations


def get_method_name_python(code):
    Node_name = ''
    for code_line in code.split("\n"):
        function = re.findall(r"def\s+(\w+)\s*\(", code_line)
        Node_name = function[0]
        break
    return Node_name

# print(get_method_name_python("def uuid2buid(value):\n    \"\"\"\n    Convert a UUID object to a 22-char BUID string\n\n    >>> u = uuid.UUID('33203dd2-f2ef-422f-aeb0-058d6f5f7089')\n    >>> uuid2buid(u)\n    'MyA90vLvQi-usAWNb19wiQ'\n    \"\"\"\n    if six.PY3:  # pragma: no cover\n        return urlsafe_b64encode(value.bytes).decode('utf-8').rstrip('=')\n    else:\n        return six.text_type(urlsafe_b64encode(value.bytes).rstrip('='))"))

# fall_method, behind_call_relations, forward_call_relations = parse_adg_python("../cosqa-train.json")
# print(json.dumps([{"fcall":forward_call_relations["transform_from_rot_trans"],"bcall":behind_call_relations["transform_from_rot_trans"]}]))
# with open("../cosqa-train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # list[dict{}]
# f.close()
# all_method, behind_call_relations,forward_call_relations=get_calling_relationship([data[15]["code"]])
# for k,v in forward_call_relations.items():
#     # if k=="transform_from_rot_trans":
#     print(k,v)
#     print("transform_from_rot_trans" in all_method)
# print(forward_call_relations["transform_from_rot_trans"])
# print([{"fcall":forward_call_relations["transform_from_rot_trans"] if "transform_from_rot_trans" in forward_call_relations.keys() else [],"bcall":behind_call_relations["transform_from_rot_trans"] if "transform_from_rot_trans" in behind_call_relations.keys() else []}])
# print(behind_call_relations["transform_from_rot_trans"])