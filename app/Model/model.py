
import pickle
import re
from pathlib import Path


__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
	mlModel = pickle.load(f)

classes =["Arabic", 
		  "Danish", 
		  "Dutch", 
		  "English", 
		  "French", 
		  "German", 
		  "Greek", 
		  "Hindi", 
		  "Italian", 
		  "Kannada", 
		  "Malayalam", 
		  "Portugeese",
		  "Russian", 
		  "Spanish", 
		  "Sweedish",		 
		  "Tamil", 
		  "Turkish"]

def Predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = mlModel.predict([text])
    return classes[pred[0]]