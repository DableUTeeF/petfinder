import numpy as np


class CSVData:
    def __init__(self, path, test_split=0.2):
        temp = open(path, 'r').readlines()
        data = []
        for line in temp[1:]:
            temp_line = line[:-1].split(',')
            t = {}
            for idx, field in enumerate(temp[0][:-1].split(',')):
                t[field] = temp_line[idx]
            data.append(t)
        self.test_split = test_split
        num_train = int((1-self.test_split)*len(data))
        self.train_data = data[:num_train]
        self.val_data = data[num_train:]


class Generator:
    def __init__(self, data):
        self.data = data
        self.current_idx = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.current_idx += 1

        y = np.array(int(self.data[idx]['AdoptionSpeed']), dtype='uint8')
        return y

    def __next__(self):
        self.current_idx += 1
        return self[self.current_idx]
