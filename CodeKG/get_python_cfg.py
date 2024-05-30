from staticfg import CFGBuilder

code = "def writeBoolean(self, n):\n        \"\"\"\n        Writes a Boolean to the stream.\n        \"\"\"\n        t = TYPE_BOOL_TRUE\n\n        if n is False:\n            t = TYPE_BOOL_FALSE\n\n        self.stream.write(t)"


def _visit_blocks(block, cfgnode, cfgedge, visited=[]):
    # Don't visit blocks twice.
    if block.id in visited:
        return cfgnode, cfgedge

    nodelabel = block.get_source()
    # print(str(block.id),nodelabel)
    # print(cfgnode)
    cfgnode.append({str(block.id): nodelabel})

    visited.append(block.id)

    # Recursively visit all the blocks of the CFG.
    for exit in block.exits:
        cfgnode, cfgedge = _visit_blocks(exit.target, cfgnode, cfgedge, visited)
        edgelabel = exit.get_exitcase().strip()
        cfgedge.append({"from": str(block.id), "to": str(exit.target.id), "condition": edgelabel})

    return cfgnode, cfgedge

def _build_cfg(cfg,cfgnode,cfgedge):
    cfgnode,cfgedge = _visit_blocks(cfg.entryblock, cfgnode,cfgedge,visited=[])
    # Build the subgraphs for the function definitions in the CFG and add
    # them to the graph.
    for subcfg in cfg.functioncfgs:

        cfgnode,cfgedge= _build_cfg(cfg.functioncfgs[subcfg],cfgnode,cfgedge)

    return cfgnode,cfgedge

def parse_cfg_python(source_code):
    cfg = CFGBuilder().build_from_src("_", source_code)
    cfcfgnode, cfgedge = [], []
    cfgnode, cfgedge = _build_cfg(cfg,cfcfgnode,cfgedge)

    return cfgnode, cfgedge

# import json
# with open("cosqa-dev.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # list[dict{}]
# f.close()
# success = 0
# fail = 0
# for dict in data:
#     print(dict["idx"])
#     try:
#         cfgnode,cfgedge=parse_cfg_python(dict["code"])
#         # print("node: ",cfgnode)
#         # print("edge: ",cfgedge)
#         success+=1
#     except:
#         print(dict["code"])
#         fail+=1
# print(f"success {success}")
# print(f"fail {fail}")
# cfgnode, cfgedge = parse_cfg_python(code)
# print(cfgnode)
# print(cfgedge)