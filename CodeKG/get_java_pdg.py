import subprocess
import json
import os
import re
import shutil
def parse_pdg_java(code):

    folder_path="../joern/joern-cli/java_pdg_results"
    os.mkdir(folder_path)
    with open("../joern/joern-cli/temp_javafile.java","w") as f:
        f.write("public class main{\n"+code+"\n}")
    f.close()
    script_path = "parse_java_pdg_by_joern.sh"
    subprocess.run(["bash", script_path])
    nodes,edges = read_pdg_result("../joern/joern-cli/java_pdg_results/pdg_output/0-pdg.dot")
    shutil.rmtree("../joern/joern-cli/java_pdg_results")
    return nodes,edges


def read_pdg_result(file):
    with open(file,"r") as f:
        dot_string = f.read()
    dot_string = re.sub(r'<SUB>[^<]+</SUB>', '', dot_string)
    node_pattern = r'"(\d+)" \[label = <([^>]*)>'
    edge_pattern = r'"(\d+)"'
    nodes = []
    edges = []
    lines = dot_string.split("\n")
    for line in lines:

        if line.startswith('"'):
            node = {}
            match = re.findall(node_pattern, line)
            if match:
                # print(match)
                node["idx"] = match[0][0]
                node["label"] = match[0][1]
                nodes.append(node)
        elif "->" in line:
            edge = {}
            match = re.findall(edge_pattern, line)
            if match:
                # print(match)
                # print(line.split("label = \"")[1].split("\"]")[0])
                edge["from"] = match[0]
                edge["to"] = match[1]
                edge["label"] = line.split("label = \"")[1].split("\"]")[0]
                edges.append(edge)

    return nodes,edges
# with open("deepcom-train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # list[dict{}]
# f.close()
# print(data[12]["code"])
# nodes,edges = parse_pdg_java(data[12]["code"])
# print(nodes)
# print(edges)