import pickle
import mudlo

with open('mudl.pkl', 'rb') as file:
    mudl = pickle.load(file)

mudl.print()
