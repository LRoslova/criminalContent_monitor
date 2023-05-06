
FROM  tensorflow/tensorflow
WORKDIR /tensorflow_model

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD python ./text_analize.py