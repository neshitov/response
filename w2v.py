import gensim
import time
from scipy.spatial.distance import cosine

slim_filename = 'GoogleNews-vectors-negative300-SLIM.bin.gz'
start = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format(slim_filename, binary=True)
print('Finished loading slim model %.1f sec' % ((time.time()-start)))
queen = model.get_vector('queen')
king = model.get_vector('king')
man = model.get_vector('man')
woman = model.get_vector('woman')
dog = model.get_vector('servant')
print('distance', cosine(dog, man+queen-woman))
