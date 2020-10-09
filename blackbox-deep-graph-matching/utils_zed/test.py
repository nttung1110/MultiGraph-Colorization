idx = 1
my_l = []
file_p = open("tmp.txt", "w")
while idx<=500:
    my_l.append(str(idx))
    idx+=1
my_l=map(lambda x:'"'+str(x)+'",\n', my_l)
file_p.writelines(my_l)
file_p.close()