base_path = "your path"
file_write = open("./data.txt",'w')
for i in range (start, end+1):
  num = str(i)
  file_input = base_path + num + ".txt"
  with open(file_input) as f:
    line = f.readline()
    while line:
      if line[1] == '4' and line[2] == '5' and line[3] == '1' and line[4] == '1':
        print line
        for data in line:
          file_write.write(data)
        line = f.readline()
        continue
      print line
      line = f.readline()
