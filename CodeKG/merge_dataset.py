import json
# file1 = "codesearchnet-train0.jsonl"
# file2 = "codesearchnet-test0.jsonl"
file = "concode-all.json"
with open(file,"w") as f :
    f.write("[")
    # for i in range(0,7):
    f1 = open("concode-train.json","r")
    sample = f1.readline()
    cnt = 0
    while(len(sample)):
        f.write(sample)
        sample = f1.readline()
        f.write(",")
    f1.close()
    f2 = open("concode-test.json", "r")
    sample = f2.readline()
    while (len(sample)):
        f.write(sample)
        f.write(",")
        sample = f2.readline()
        # if len(sample):

    f2.close()
    f3 = open("concode-valid.json", "r")
    sample = f3.readline()
    while (len(sample)):
        f.write(sample)
        sample = f3.readline()
        if len(sample):
            f.write(",")
    f3.close()
    f.write("]")
f.close()