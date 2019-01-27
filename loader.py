import joblib
#import mudlo

with open('mudl.pkl', 'rb') as file:
    mudl = joblib.load(file)

mudl.print()
