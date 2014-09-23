from glob import glob
from posix import mkdir, chdir
from os import rename, path

TRAIN_SIZE = 300
DEV_SIZE = 30
# TEST on all the rest

def split_passages():
    chdir('passages')
    passages = glob('*.xml')
    for split in 'train', 'dev', 'test':
        if not path.exists(split):
            mkdir(split)
    for f in passages[:TRAIN_SIZE]:
        rename(f, 'train/'+f)
    for f in passages[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE]:
        rename(f, 'dev/'+f)
    for f in passages[TRAIN_SIZE+DEV_SIZE:]:
        rename(f, 'test/'+f)

if __name__ == '__main__':
    split_passages()