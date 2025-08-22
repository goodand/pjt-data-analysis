import chardet

with open('file.txt', 'rb') as f:
    raw = f.read()
print(chardet.detect(raw))