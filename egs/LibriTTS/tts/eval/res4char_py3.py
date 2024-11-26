import sys
import re
# python eval/res4char_py3.py testsetA/test_text.scp >> testsetA/test_text_norm.scp
def scoreformat(name, line):
    newline = ""
    lastEn = False
    for i in range(0, len(line)):
        curr = line[i]
        currEn = False
        if curr == "":
            continue
        if curr.encode("utf8").upper() >= b'A' and curr.encode("utf8").upper() <= b'Z' or curr == '\'':
            currEn = True
        if i == 0:
            newline = newline + curr.lower()
        else:
            if lastEn == True and currEn == True:
                newline = newline + curr.lower()
            else:
                newline = newline + " " + curr.lower()
        lastEn = currEn
    ret = re.sub("[ ]{1,}", ' ', newline)
    ret = name + "\t" + ret
    return ret

def recoformat(line):
    newline = ""
    en_flag = 0   # 0: no-english   1 : english   2: former 
    for i in range(0, len(line)):
        word = line[i]
        if ord(word) == 32:
            if en_flag == 0:
                continue
            else:
                en_flag = 0
                newline += " "
        if (word >= '一' and word <= '龥') or (word >= '0' and word <= '9'):
            if en_flag == 1:
                newline += " " + word
            else:
                newline += word
            en_flag = 0
        elif (word >= 'A' and word <= 'Z') or (word >= 'a' and word <= 'z') or word == '\'':
            if en_flag == 0:
                newline += " " + word
            else:
                newline += word
            en_flag = 1
        else:
            newline += " "
    newline = re.sub("[ ]{1,}", ' ', newline)
    newline = newline.strip()
    return newline

if __name__ == '__main__':
    if len(sys.argv[1:]) < 1:
        sys.stderr.write("Usage:\n .py  reco.result\n")
        sys.stderr.write(" reco.result:   id<delimiter>recoresult; delimiter: \\t , blank \n")
        sys.exit(1)
    
    f = open(sys.argv[1])
    for line in f:
        if not line:
            continue
        line = line.rstrip()
        if '\t' in line:
            tmp = line.split('\t')
        elif ',' in line:
            tmp = line.split(',', 1)
        else:
            tmp = line.split(' ', 1)
        if len(tmp) < 2:
            continue
        name = tmp[0]
        content = tmp[1]
        content = recoformat(content)
        content = scoreformat(name, content)
        print(content)
    f.close()
