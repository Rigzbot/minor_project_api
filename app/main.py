from fastapi import FastAPI, HTTPException, File, UploadFile
from app.yolov8 import YOLOv8_face
import cv2
from app.eval_spotify_annoy import get_face_enrollement_number
import shutil
from fastapi.responses import JSONResponse
import uuid
import os

# config
yolo_model_path = "app/weights/yolov8n-face.onnx"
conf_threshold = 0.45
nmsthreshold = 0.5
detected_faces_directory = "app/unknown_faces"
IMAGEDIR = "app/fastapi-images/"

app = FastAPI()

#Upload file to server
@app.post("/images")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        contents = await file.read()  # <-- Important!

        os.makedirs(IMAGEDIR, exist_ok=True)
        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)

        return JSONResponse(content={"filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get list of faces in uploaded image
@app.get("/images")
async def get_enrollement_number():
    try:
        files = os.listdir(IMAGEDIR)
        img_path = f"{IMAGEDIR}{files[0]}"

        src_img = cv2.imread(img_path)

        YOLOv8_face.detect_faces(model_path=yolo_model_path, 
                             nmsThreshold=nmsthreshold, 
                             confThreshold=conf_threshold, 
                             srcimg=src_img, 
                             detected_faces_directory=detected_faces_directory)
    
        result = get_face_enrollement_number(detected_faces_directory)

        #Delete directory of detected faces
        shutil.rmtree(detected_faces_directory)
        shutil.rmtree(IMAGEDIR)

        return JSONResponse(content={"enrollement_number_list": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn model_api:app --reload 

#Can use railways



    


