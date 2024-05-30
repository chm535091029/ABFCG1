import javalang
import json
from tqdm import tqdm
import collections
import sys
from collections import OrderedDict

import re
class Block:
    def __init__(self, code_snippet,id, in_condition = ""):
        self.current_block_code = code_snippet
        self.next_block = OrderedDict()
        self.id = id
        self.in_condition = in_condition
    def add_next_block(self,Block,out_condition):
        self.next_block[Block.id] = out_condition
        # self.out_condition.append(out_condition)
        # self.next_block = list(set(self.next_block))

def get_inner_code(code_line,start,end):
    stack = []

    i = start
    while i<end:
        for l in code_line[i]:
            if l == '{':
                if not stack:
                    start = i
                stack.append(l)
            elif l == '}':
                stack.pop()
                if not stack:
                    end = i
                    return start+1, end
        i+=1
    return None, None

def get_condition(line):
    stack = []
    start = None
    end = None

    for i, c in enumerate(line):
        if c == '(':
            if not stack:
                start = i + 1
            stack.append(c)
        elif c == ')':
            stack.pop()
            if not stack:
                end = i
                return line[start:end]
    return ""
def parse_cfg_java_re(code_lines,blocks,first,start,end,return_idx,in_idx,has_branch=False):


    back_idx = [] #记录当前访问范围内的for,while的基本块的id，用于连接给当前访问范围后面的基本块
    else_idx = [] #记录else语句的基本块id
    if_idx = [] #记录if和else if 语句的基本块id
    series_idx = [] #记录顺序执行的基本块id
    first_block = True #记录是否是本次访问第一个遇到的基本块，如果是的话特殊处理，把第一个基本块和递归进来的in_idx号代码块连起来

    if first:
        block = Block(code_lines[0],len(blocks))
        blocks.append(block)
        start, end = get_inner_code(code_lines,start,end)
        back_idx = [block.id]
    cur = start

    start1, end1 = start, end

    while cur>=start and cur<end:
        if (re.search(r'\bwhile\b', code_lines[cur]) or re.search(r'\bfor\b', code_lines[cur]) or re.search(r'\bif\b', code_lines[cur]) or re.search(r'\btry\b', code_lines[cur]) or re.search(r'\belif\b', code_lines[cur]) or re.search(r'else\s*if ', code_lines[cur]) or re.search(r'\bcatch\b', code_lines[cur])):
            has_branch = True
            if cur>start:
                block = Block("\n".join(code_lines[start:cur]), len(blocks))

                if (len(back_idx) > 0 or len(if_idx)>0 or len(series_idx)>0 or len(else_idx)>0):
                    for id in back_idx:
                        blocks[id].add_next_block(block, "")
                    for id in if_idx:
                        blocks[id].add_next_block(block, "")
                    for id in series_idx:
                        blocks[id].add_next_block(block, "")
                    for id in else_idx:
                        blocks[id].add_next_block(block, "")
                    back_idx = []
                    if_idx = []
                    else_idx = []

                if first_block:
                    blocks[in_idx].add_next_block(block, blocks[in_idx].in_condition)
                    first_block = False
                series_idx = [block.id] #对于顺序语句的块连接完之前的块之后就把它们的id去掉换成自己的
                for code_line in code_lines[start:cur]:
                    if re.search(r'\breturn\b', code_line):
                        return_idx.append(len(blocks))
                        else_idx = []
                        back_idx = []
                        if_idx = []
                        series_idx = []
                        break
                blocks.append(block)
            start1, end1 = get_inner_code(code_lines,cur,end)
            if start1 is None or end1 is None:
                start1=end1=cur
                block = Block(code_lines[cur],len(blocks),get_condition(code_lines[cur]))
            else:
                block = Block("\n".join(code_lines[start1:end1]), len(blocks),get_condition(code_lines[cur]))
            if first_block:
                blocks[in_idx].add_next_block(block, get_condition(code_lines[cur]))
                first_block = False
            if len(back_idx)>0 or len(else_idx) or len(series_idx) or len(if_idx):
                for id in back_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                for id in else_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                for id in if_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                for id in series_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                # else_idx = []
            if re.search(r'\bfor\b', code_lines[cur]) or re.search(r'\bwhile\b', code_lines[cur]):
                back_idx.append(block.id)
            elif re.search(r'\bif\b', code_lines[cur]) or re.search(r'\btry\b', code_lines[cur]) or re.search(r'\belif\b', code_lines[cur]) or re.search(r'else\s*if ', code_lines[cur]) or re.search(r'\bcatch\b', code_lines[cur]):
                if_idx.append(block.id)

            for code_line in code_lines[start1:end1]:
                if re.search(r'\breturn\b', code_line):
                    return_idx.append(block.id)

                    break
            blocks.append(block)
            for code_line in code_lines[start1:end1]:
                if re.search(r'\bwhile\b', code_line) or re.search(r'\bfor\b', code_line) or re.search(r'\bif\b', code_line) or re.search(r'\btry\b', code_line):
                    blocks,return_idx = parse_cfg_java_re(code_lines,blocks,False,start1,end1,return_idx,block.id)
                    break
            # print(f"start1:{start1}")
            # print(f"end1:{end1}")
            cur = end1
            start = cur + 1
        # elif re.search(r'\belse\b', code_lines[cur]) or re.search(r'\belif\b', code_lines[cur]) or re.search(r'else\s*if ', code_lines[cur]):
        elif re.search(r'\belse\b', code_lines[cur]):
            has_branch = True
            start1, end1 = get_inner_code(code_lines, cur, end)
            if start1 is None or end1 is None:
                start1=end1=cur
                block = Block(code_lines[cur],len(blocks),get_condition(code_lines[cur]))
            else:
                block = Block("\n".join(code_lines[start1:end1]), len(blocks),get_condition(code_lines[cur]))
            if len(back_idx) > 0 or len(series_idx)>0 or len(if_idx):
                for id in back_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                back_idx = []
                for id in series_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))
                series_idx = []
                for id in if_idx:
                    blocks[id].add_next_block(block, get_condition(code_lines[cur]))

            else_idx=[block.id]
            blocks.append(block)

            for code_line in code_lines[start1:end1]:
                if re.search(r'\bwhile\b', code_line) or re.search(r'\bfor\b', code_line) or re.search(r'\bif\b', code_line) or re.search(r'\btry\b', code_line):
                    blocks,return_idx = parse_cfg_java_re(code_lines,blocks,False,start1,end1,return_idx,block.id)
                    break
            cur = end1
            start = cur + 1
            # print(f"start1:{start1}")
            # print(f"end1:{end1}")
        elif re.search(r'\bswitch\b', code_lines[cur]): #不对switch进行更细致处理，只是将整体视为一个块
            has_branch = True
            start1, end1 = get_inner_code(code_lines, cur, end)
            if start1 is None or end1 is None:
                start1=end1=cur
                block = Block(code_lines[cur],len(blocks),get_condition(code_lines[cur]))
            else:
                block = Block("\n".join(code_lines[start1:end1]), len(blocks),get_condition(code_lines[cur]))
            if (len(back_idx) > 0 or len(if_idx) > 0 or len(series_idx) > 0 or len(else_idx)>0):
                for id in back_idx:
                    blocks[id].add_next_block(block, "")
                for id in if_idx:
                    blocks[id].add_next_block(block, "")
                for id in series_idx:
                    blocks[id].add_next_block(block, "")
                for id in else_idx:
                    blocks[id].add_next_block(block, "")

                back_idx = []
                if_idx = []
                else_idx = []
            series_idx = [block.id]
            cur = end1
            start = cur + 1
        cur += 1
        if cur==end:
            flag = True
            if  has_branch==False:
                block = Block("\n".join(code_lines[start:end1]), len(blocks))
            elif has_branch and end1+1<end:
                block = Block("\n".join(code_lines[end1+1:end]), len(blocks))
            else:
                flag = False
            if flag:
                # 如果有顺序的代码块就把这个代码块和之前所有的分支语句连接起来
                if (len(back_idx) > 0 or len(if_idx) > 0 or len(series_idx) > 0):
                    for id in back_idx:
                        blocks[id].add_next_block(block, "")
                    for id in if_idx:
                        blocks[id].add_next_block(block, "")
                    for id in series_idx:
                        blocks[id].add_next_block(block, "")
                    else_idx = []
                    back_idx = []
                    if_idx = []

                if len(else_idx):
                    for id in else_idx:
                        blocks[id].add_next_block(block, "")
                    else_idx = []
                series_idx.append(block.id)
                if re.search(r'\breturn\b', block.current_block_code):
                    return_idx.append(len(blocks))
                    else_idx = []
                    back_idx = []
                    if_idx = []
                    series_idx = []
                blocks.append(block)
            if first:
                exit_block = Block("exit", len(blocks))
                blocks.append(exit_block)
                if flag:
                    blocks[block.id].add_next_block(blocks[exit_block.id], get_condition(code_lines[cur]))
                if len(return_idx) > 0:
                    for id in return_idx:
                        if block.id!=id:
                            blocks[id].add_next_block(blocks[exit_block.id], get_condition(code_lines[cur]))
                if (len(back_idx)>0 or len(else_idx) or len(if_idx) or len(series_idx)) and flag is False:
                    for id in back_idx:
                        blocks[id].add_next_block(exit_block, get_condition(code_lines[cur]))
                    for id in else_idx:
                        blocks[id].add_next_block(exit_block, get_condition(code_lines[cur]))
                    for id in if_idx:
                        blocks[id].add_next_block(exit_block, get_condition(code_lines[cur]))
                    for id in series_idx:
                        blocks[id].add_next_block(exit_block, get_condition(code_lines[cur]))
            elif first is False:

                if flag:
                    blocks[block.id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                # if len(return_idx) > 0:
                #     for id in return_idx:
                #         if block.id!=id:
                #             # blocks[id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                #             blocks[id].add_next_block(blocks[exit_block.id], get_condition(code_lines[cur]))
                if (len(back_idx)>0 or len(else_idx) or len(if_idx) or len(series_idx)) and flag is False:
                    for id in back_idx:
                        blocks[id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                    for id in else_idx:
                        blocks[id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                    for id in if_idx:
                        blocks[id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                    for id in series_idx:
                        blocks[id].add_next_block(blocks[in_idx], get_condition(code_lines[cur]))
                    # blocks[id].add_next_block(blocks[exit_block.id],get_condition(code_lines[cur]))

    return blocks,return_idx
# blocks = []
# start1 = end1 = -1
# blocks = parse_cfg_java_re(code.split("\n"),blocks,True,0,len(code.split("\n")))
code = """public static void main(String[] args) {
    int i = 0;
    int k = i+j:

    i++;
    if (k>1000){
        c++;
    }
    else{
        k++;
        }
    k--;
    if(y==1){
        y++;
    }
    else{
        y--
        }
}
"""
# code = """public static void main(String[] args) {
#     int i = 0;
#     int k = i+j:
#     while (i < 10) {
#         int j = 0;
#         while (j < 5) {
#             System.out.println("i = " + i + ", j = " + j);
#             j++;
#         }
#         i++;
#         if (k>1000){
#             c++;
#         }
#         else{
#             k++;
#             }
#         j--:
#     }
# }
# """
# code = """@KnownFailure("Fixed on DonutBurger, Wrong Exception thrown") public void test_unwrap_ByteBuffer_ByteBuffer_02(){
#   String host="new host";
#   int port=8080;
#   ByteBuffer bbs=ByteBuffer.allocate(10);
#   ByteBuffer bbR=ByteBuffer.allocate(100).asReadOnlyBuffer();
#   ByteBuffer[] bbA={bbR,ByteBuffer.allocate(10),ByteBuffer.allocate(100)};
#   SSLEngine sse=getEngine(host,port);
#   sse.setUseClientMode(true);
#   try {
#     sse.unwrap(bbs,bbA);
#     fail("ReadOnlyBufferException wasn't thrown");
#   }
#  catch (  ReadOnlyBufferException iobe) {
#   }
# catch (  Exception e) {
#     fail(e + " was thrown instead of ReadOnlyBufferException");
#   }
# }
# """

# code = """private byte[] calculateUValue(byte[] generalKey,byte[] firstDocIdValue,int revision) throws GeneralSecurityException, EncryptionUnsupportedByProductException {
#   if (revision == 2) {
#     Cipher rc4=createRC4Cipher();
#     SecretKey key=createRC4Key(generalKey);
#     initEncryption(rc4,key);
#     return crypt(rc4,PW_PADDING);
#   }
#  else   if (revision >= 3) {
#     MessageDigest md5=createMD5Digest();
#     md5.update(PW_PADDING);
#     if (firstDocIdValue != null) {
#       md5.update(firstDocIdValue);
#     }
#     final byte[] hash=md5.digest();
#     Cipher rc4=createRC4Cipher();
#     SecretKey key=createRC4Key(generalKey);
#     initEncryption(rc4,key);
#     final byte[] v=crypt(rc4,hash);
#     rc4shuffle(v,generalKey,rc4);
#     assert v.length == 16;
#     final byte[] entryValue=new byte[32];
#     System.arraycopy(v,0,entryValue,0,v.length);
#     System.arraycopy(v,0,entryValue,16,v.length);
#     return entryValue;
#   }
#  else {
#     throw new EncryptionUnsupportedByProductException("Unsupported standard security handler revision " + revision);
#   }
# }"""
# print(code)
# blocks ,_= parse_cfg_java_re(code.split("\n"),blocks,True,0,len(code.split("\n")),[],0)
# with open("deepcom-train.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # list[dict{}]
# f.close()
# # print(data[12]["code"])
# blocks, _ = parse_cfg_java_re(data[12]["code"].split("\n"),blocks,True,0,len(data[12]["code"].split("\n")),[],0)
# for b in blocks:
#     print(str(b.id)+":"+b.current_block_code)
#     print("next" + ":" + str(b.next_block))
def parse_cfg_java(code):
    # code_lines,blocks,first,start,end,return_idx,in_idx,has_branch=False
    blocks, _ = parse_cfg_java_re(code.split("\n"),[],True,0,len(code.split("\n")),[],0)
    cfg_dicts = []
    for b in blocks:
        cfg_dict = {}
        cfg_dict["id"] = b.id
        cfg_dict["block_code"] = b.current_block_code
        cfg_dict["next_blocks"] = b.next_block
        cfg_dicts.append(cfg_dict)
    return cfg_dicts
print(parse_cfg_java(code))
import json
