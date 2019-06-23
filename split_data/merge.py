import os

dirlist = [item for item in os.listdir('./') if os.path.isdir(item)]

# my own defined filter
dirlist = [item for item in dirlist if 'month' in item]

for dirp in dirlist:
    filelist = os.listdir(dirp)
    with open(dirp + '/' + os.path.basename(dirp) + '.tsv', 'w') as writef:
        with open(dirp + '/' + filelist[0]) as dataf:
            writef.write(dataf.readline()) # column names

        for filep in filelist:
            print(filep)
            with open(dirp + '/' + filep) as dataf:
                dataf.readline()
                for line in dataf:
                    writef.write(line)
