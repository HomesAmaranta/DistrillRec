import fire
import collections

# 生成数据集，格式：user item1 item2...


def run(path: str = "./Beauty.txt"):
    res = ""
    dic = collections.OrderedDict()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            user = line[0]
            item = line[1]
            if user in dic:
                dic[user].append(item)
            else:
                dic[user] = [item]

    for key, value in dic.items():
        res += str(key)
        for v in value:
            res += " "+str(v)
        res += "\n"

    with open(path[:-4]+"_processed.txt", 'w') as f:
        f.write(res)
    print("Done.")

# 找出物品索引的范围;Beauty数据集中，只有0未出现，所以范围是1~item_num
def run2(path: str = "./Beauty.txt"):
    dic = dict()
    max = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            item = int(line[1])
            if item > max:
                max = item
            if item not in dic:
                dic[item] = 1
    for i in range(max+1):
        if i not in dic:
            print(i)

    print("Done.")


if __name__ == "__main__":
    fire.Fire(run2)
