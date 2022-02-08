students = [student.split('@')[0] for student in open('../../../students').readlines()]

def readConll(path):
    curSent = []
    allData = []
    for line in open(path):
        if len(line) < 2:
            allData.append(curSent)
            curSent = []
        else:
            curSent.append(line.split('\t')[0])
    return allData

data = readConll('da_arto.conll')
print(len(data))
print(len(students))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


parts = list(chunks(data, 20))
for studentIdx, student in enumerate(students):
    partIdx = int(studentIdx/4)
    outFile = open('pos-' + student + '-' + str(partIdx) + '.conll', 'w')
    for sent in parts[partIdx]:
        for word in sent:
            outFile.write(word + '\t\n')
        outFile.write('\n')
    outFile.close()





