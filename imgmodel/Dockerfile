
FROM  tensorflow/tensorflow
WORKDIR /model

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD python ./img_analize.py