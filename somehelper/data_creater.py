# получим объект файла
file1 = open("./database/alcohol_drugs.txt", "r")

for i in range(1,501):
    line = file1.readline()
    if i<401:
        file_name = './dataframe/train/alcohol_drugs/1_{}.txt'.format(i)
        with open(file_name,'w') as f:
            # res += str(i)
            f.write(line)
    else:
        file_name = './dataframe/test/alcohol_drugs/1_{}.txt'.format(i)
        with open(file_name,'w') as f:
            # res += str(i)
            f.write(line)


file1.close