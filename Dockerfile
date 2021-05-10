FROM python:3.8

WORKDIR /digit-recognition-api-app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./src ./src

COPY ./data ./data

WORKDIR ./src

CMD ["python", "app.py"]
