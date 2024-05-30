import threading
import subprocess
import os
import json
from tqdm import tqdm
import re
max_threads = 8
thread_semaphore = threading.Semaphore(max_threads)

def make_javafile(dataset,save_position):
    with open(dataset, "r", encoding="utf-8") as r:
        data = json.load(r)
        os.mkdir(os.path.join(save_position, dataset.split("/")[-1].split(".")[0]))
        for i, dict in tqdm(enumerate(data), desc="creating java files"):
            file_path = os.path.join(save_position, dataset.split("/")[-1].split(".")[0],  str(i) + ".java")
            with open(file_path, "w") as w:
                # print(file_path)
                w.write("public class main{\n" + dict["code"] + "\n}")
            w.close()
    r.close()
def process_file2cpg(java_dir,filename, out_path):
    # 获取信号量，控制线程数
    with thread_semaphore:
        dir_name = filename.split(".")[0]
        try:
            subprocess.run(['../joern/joern-cli/joern-parse', os.path.join(java_dir ,filename), "--output", out_path+"/"+dir_name+"_cpg"])
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

def process_file2pdg(filename, out_path):
    with thread_semaphore:
        dir_name = filename.split(".")[0]
        try:
            subprocess.run(['../joern/joern-cli/joern-export', out_path+"/"+dir_name+"_cpg", '--repr', 'pdg', '--out', out_path+"/"+dir_name])
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
def parse_pdg_java2file(java_dir, output_dir):
    files_list = os.listdir(java_dir)
    threads = []
    for file in tqdm(files_list,desc="parsing java files to cfg files"):
        thread = threading.Thread(target=process_file2cpg, args=(java_dir,file, output_dir))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    threads = []
    for file in tqdm(files_list,desc="parsing java files to pdg files"):
        thread = threading.Thread(target=process_file2pdg, args=(file, output_dir))
        threads.append(thread)
        thread.start()


    for thread in threads:
        thread.join()


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


