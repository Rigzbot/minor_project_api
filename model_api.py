from fastapi import FastAPI, HTTPException
from main import YOLOv8_face
import numpy as np
import cv2
from eval_spotify_annoy import get_face_enrollement_number
import shutil
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# config
yolo_model_path = "weights/yolov8n-face.onnx"
conf_threshold = 0.45
nmsthreshold = 0.5
detected_faces_directory = "unknown_faces"

app = FastAPI()

class ImageData(BaseModel):
    image: str

# Create post API endpoint
@app.post("/get_enrollement_number")
async def get_enrollement_number(data: ImageData):
    try:
        # Read the uploaded image directly with OpenCV
        img_path = data.image
        src_img = cv2.imread(img_path)
        # image_data = await image_upload.image.read()
        # nparr = np.frombuffer(image_data, np.uint8)
        # src_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        YOLOv8_face.detect_faces(model_path=yolo_model_path, 
                             nmsThreshold=nmsthreshold, 
                             confThreshold=conf_threshold, 
                             srcimg=src_img, 
                             detected_faces_directory=detected_faces_directory)
    
        result = get_face_enrollement_number(detected_faces_directory)

        #Delete directory of detected faces
        shutil.rmtree(detected_faces_directory)

        return JSONResponse(content={"enrollement_number_list": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn model_api:app --reload 

#TODO(Install docker for windows, create docker file, create requirements.txt file, add redundant files in docker_ignore,)

#Can use railways



    


