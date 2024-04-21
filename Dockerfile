FROM tetsuyahhh/onnxruntime-gpu:latest
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./ /app

WORKDIR /app

COPY ./entrypoint.sh /
ENTRYPOINT ["sh", "/entrypoint.sh"]

 