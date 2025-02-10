from fastapi import FastAPI, File, UploadFile
import uvicorn
from roboflow import Roboflow
import shutil

app = FastAPI()

# Initialize Roboflow
rf = Roboflow(api_key="oYrMrgEdZhqnTa6aYrZR")
project = rf.workspace().project("damaged-books-rqxs4")
model = project.version(2).model

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.predict(file.filename, confidence=3, overlap=4)
    
    return {"message": "Prediction complete", "result": result.json()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
