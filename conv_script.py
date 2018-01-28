import os

ff = open("embedz_deep_test", 'w')
i = 1

ff.write('2708' + ' ' + str(256) + '\n')

with open('embedz') as f:
 content = f.readlines()
 for strings in content:
  strings=str(i)+' '+strings
  ff.write(strings)
  i+=1

ff.close()
