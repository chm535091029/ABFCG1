import re
from tqdm import tqdm
import json


def file_dev(dataset):
    program = []
    with open(dataset,"r",encoding="utf-8") as f:
        data = json.load(f)
        for dict in data:

            program.append(dict["code"].replace('\n', '$'))

    return program


# def build_instant_neighbour_dict(node_list, front_call, behind_call):
#     front_call_dict = {}
#     behind_call_dict = {}
#     for i in tqdm(range(len(node_list)),desc="build_instant_neighbour_dict"):
#         front_call_dict[node_list[i]] = front_call[i]
#         behind_call_dict[node_list[i]] = behind_call[i]
#     return front_call_dict, behind_call_dict


# def build_instant_neighbour(call_relations):
#     node_list = []
#     front_call = []
#     behind_call = []
#     fcall = []
#
#     for i in tqdm(call_relations,desc="behind_call"):
#         node_list.append(i[0])
#         behind_call.append(i[1:])
#     for i in tqdm(range(len(node_list)),desc="front_call"):
#         for j in range(len(node_list)):
#             if node_list[i] in call_relations[j][1:]:
#                 fcall.append(call_relations[j][0])
#         front_call.append(fcall)
#         fcall = []
#
#     return node_list, front_call, behind_call

def get_calling_relationship(program):
    func_name = []
    behind_call_relations = {}
    forward_call_relations = {}
    for i in tqdm(program,desc="get_calling_relationship"):
        # print(i)
        Func_name = re.findall(' (\w+?)\(', i.split('$')[0])
        # print(Func_name)
        #
        count = 0
        global Node_name
        if len(Func_name) > 0:
            for Fname in Func_name:
                Fname = re.sub(r'[^\w]', ' ', Fname)
                if len(list(filter(None, Fname.split(' ')))) == 1 and not Fname.replace(' ', '').isdigit() and not 'if' == Fname.replace(' ', '') and not 'for' == Fname.replace(' ', '') and not 'while' == Fname.replace(' ', ''):
                    if type(Fname) == str:
                        if count==0:
                            Node_name = Fname

                            if Fname not in behind_call_relations.keys():
                                behind_call_relations[Node_name] = []

                        else:
                            behind_call_relations[Node_name].append(Fname)
                            if Fname in forward_call_relations.keys():
                                forward_call_relations[Fname].append(Node_name)
                            else:
                                forward_call_relations[Fname] = [Node_name]
                        count+=1
                        func_name.append(Fname.replace(' ', ''))


        else:
# @KnownFailure("Fixed on DonutBurger, Wrong Exception thrown") public void test_unwrap_ByteBuffer_ByteBuffer_02(){$
            Fname = i.split('$')[0].split('(')[0]

            if "@" not in Fname:
                if type(Fname)==str:
                    if count == 0:
                        Node_name = Fname
                        if Fname not in behind_call_relations.keys():
                            behind_call_relations[Node_name] = []

                    else:
                        behind_call_relations[Node_name].append(Fname)
                        if Fname in forward_call_relations.keys():

                            forward_call_relations[Fname].append(Node_name)
                        else:
                            forward_call_relations[Fname] = [Node_name]
                    count += 1
                    func_name.append(Fname.replace(' ', ''))

            else:
                for part in i.split('$'):
                    Fname = re.findall(' (\w+?)\(', part)
                    if len(Fname)>0:
                        for name in Fname:
                            if type(name) == str:
                                if count == 0:
                                    # print(1)
                                    Node_name = name
                                    if name not in behind_call_relations.keys():
                                        behind_call_relations[Node_name] = []

                                else:  #
                                    behind_call_relations[Node_name].append(name)
                                    if name in forward_call_relations.keys():
                                        forward_call_relations[name].append(Node_name)
                                    else:
                                        forward_call_relations[name] = [Node_name]
                                count += 1
                                func_name.append(name.replace(' ', ''))
                                break;


        for n,sentence in enumerate(i.split('$')):
            if n!=0:

                fnames = re.findall('(\w+?)\(', sentence)
                if len(fnames) > 0:
                    for fname in fnames:

                        fname = re.sub(r'[^\w]', ' ', fname)
                        if len(list(filter(None, fname.split(' ')))) > 1 or fname.replace(' ', '').isdigit() or 'if' == fname.replace(' ', '') or 'for' == fname.replace(' ', '') or 'while' == fname.replace(' ',''):
                            continue
                        else:
                            if type(fname) == str:
                                if count == 0:
                                    Node_name = fname
                                    if fname not in behind_call_relations.keys():
                                        behind_call_relations[Node_name] = []
                                else:
                                    behind_call_relations[Node_name].append(fname)
                                    if fname in forward_call_relations.keys():
                                        # print(forward_call_relations[Fname])
                                        forward_call_relations[fname].append(Node_name)
                                    else:
                                        forward_call_relations[fname] = [Node_name]
                                count += 1
                                func_name.append(fname.replace(' ', ''))


    return list(filter(None, list(set(func_name)))), behind_call_relations,forward_call_relations
def parse_adg_java(filename):
    code = file_dev(filename)
    all_method, behind_call_relations,forward_call_relations = get_calling_relationship(code)
    # node_list, front_call, behind_call = build_instant_neighbour(call_relations)
    # front_call_dict, behind_call_dict = build_instant_neighbour_dict(node_list, front_call, behind_call)
    return all_method, behind_call_relations,forward_call_relations

def get_method_name_java(code):
    Func_name = re.findall(' (\w+?)\(', code.split('$')[0])
    # print(Func_name)
    #

    Node_name = ''
    if len(Func_name) > 0:

        Func_name[0] = re.sub(r'[^\w]', ' ', Func_name[0])

        if len(list(filter(None, Func_name[0].split(' ')))) == 1 and not Func_name[0].replace(' ',
                                                                                '').isdigit() and not 'if' == Func_name[0].replace(
                ' ', '') and not 'for' == Func_name[0].replace(' ', '') and not 'while' == Func_name[0].replace(' ', ''):
            if type(Func_name[0]) == str:
                Node_name = Func_name[0]

    else:
        # @KnownFailure("Fixed on DonutBurger, Wrong Exception thrown") public void test_unwrap_ByteBuffer_ByteBuffer_02(){$
        Fname = code.split('$')[0].split('(')[0]

        if "@" not in Fname:
            if type(Fname) == str:
                Node_name = Fname

        else:
            for part in code.split('$'):
                Fname = re.findall(' (\w+?)\(', part)
                if len(Fname) > 0:
                    if type(Fname[0]) == str:

                        Node_name =Fname[0]
                        break
    return Node_name

# print(get_method_name_java("@KnownFailure(\"Fixed on DonutBurger, Wrong Exception thrown\") public void test_unwrap_ByteBuffer_ByteBuffer_02(){\n  String host=\"new host\";\n  int port=8080;\n  ByteBuffer bbs=ByteBuffer.allocate(10);\n  ByteBuffer bbR=ByteBuffer.allocate(100).asReadOnlyBuffer();\n  ByteBuffer[] bbA={bbR,ByteBuffer.allocate(10),ByteBuffer.allocate(100)};\n  SSLEngine sse=getEngine(host,port);\n  sse.setUseClientMode(true);\n  try {\n    sse.unwrap(bbs,bbA);\n    fail(\"ReadOnlyBufferException wasn't thrown\");\n  }\n catch (  ReadOnlyBufferException iobe) {\n  }\ncatch (  Exception e) {\n    fail(e + \" was thrown instead of ReadOnlyBufferException\");\n  }\n}\n"))