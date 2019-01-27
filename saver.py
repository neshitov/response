import pickle
from mudlo import Mudlo

mudl = Mudlo()

with open('mudl.pkl', 'wb') as file:
    pickle.dump(mudl, file)
