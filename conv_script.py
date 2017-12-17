import os

class Conversion:
    def __init__(self):
        pass

    def converter(self, normal_embeddings_file, deep_embeddings_file, embed_dims):

        ff = open(deep_embeddings_file, 'w')
        i = 0

        ff.write('10312' + ' ' + str(embed_dims) + '\n')

        with open(normal_embeddings_file) as f:
            content = f.readlines()
            for strings in content:
                strings=str(i)+' '+strings
                ff.write(strings)
                i+=1

        ff.close()