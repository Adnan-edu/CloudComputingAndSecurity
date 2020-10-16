FROM ubuntu:18.04
RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev
COPY ./requirements.txt /adnan/requirements.txt
WORKDIR /adnan
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . /adnan
CMD ["python3", "serveradnan.py" ]
