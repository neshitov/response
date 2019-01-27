import joblib
from mudlo import Mudlo

mudl = Mudlo()

with open('mudl.pkl', 'wb') as file:
    joblib.dump(mudl, file)
