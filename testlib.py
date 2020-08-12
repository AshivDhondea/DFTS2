import yaml
import sys
import os

testdir = 'largeTest'

print(testdir)

directory = os.path.dirname(__file__)
testdir_clean = os.path.join(directory,testdir)

for file in os.listdir(testdir_clean):
    if file.endswith(".txt"):
        print(os.path.join(testdir_clean, file))
