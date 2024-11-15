from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, path, maxlen=50):
        super(SeqDataset, self).__init__()

        self.train_data = []
        self.test_data = {}
        self.val_data = []
        self.item_max=0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(t)-1 for t in line[1:]]#物品是从1开始编号，为了使用CrossEntropyLoss函数（标签从0开始），所以都减1
                self.item_max=max(self.item_max,max(items))
                length = len(items)
                if length >= 3:
                    # self.val_data[user] = [items[:-2], items[-2]]
                    self.val_data.append([items[:-2], items[-2]])
                    self.test_data[user] = [items[:-1], items[-1]]
                for t in range(1, length):
                    self.train_data.append(
                        [items[:-length+t], items[-length+t]])

    def __getitem__(self, idx):
        seq, label = self.train_data[idx]
        return seq, label

    def __len__(self):
        return len(self.train_data)
