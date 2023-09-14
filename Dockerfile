FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install uvicorn
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -t face_recognition_app .  --> Used to build the docker file, face_recognition_app is name of docker file created

# docker run -d --name mycontainer -p 80:80 face_recognition_app --> Used to run the created docker file