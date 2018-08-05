import pickle

data = dict()
dictPath = './deeplab-pytorch/test_images/test.txt'

with open(dictPath, 'rb') as file:
    data = pickle.load(file, fix_imports=True, encoding="ASCII", errors="strict")

print(type(data))
