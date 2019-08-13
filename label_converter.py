import os
from shutil import copyfile

link = os.path.join('..', 'crkotina')
output = os.path.join('..', 'crkotinaUpdate')

for filename in os.listdir(os.path.join(link, '')):
    if filename.endswith(".txt"):
        f = open(os.path.join(link, f"{filename}"), "r")
        txt = f.read().split('\t')
        fajl = txt[0]
        sx = int(txt[1])
        sy = int(txt[2])
        ex = int(txt[3])
        ey = int(txt[4])
        w=ex-sx
        h=ey-sy

        out = open(os.path.join(output,f"{filename}"),"w+")
        out.write(str(fajl)+'\t'+str(sx)+'\t'+str(sy)+'\t'+str(w)+'\t'+str(h)+'\n')
        copyfile(os.path.join(link,fajl),os.path.join(output,fajl))