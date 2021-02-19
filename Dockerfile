# a simple docker example
From python:3.7
WORKDIR /app
COPY . /
ADD . /app

RUN pip install requests
RUN pip install numpy
RUN pip install tensorflow
RUN pip install keras
RUN pip install sklearn
RUN pwd

ENTRYPOINT [ "python", "TF_Aggregate.py" ]
