FROM python:3.8-slim

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

# prefer GPU support? use the other docker file.

COPY . /app
WORKDIR /app

RUN chmod 755 ./gunicorn.sh
ENTRYPOINT ["./gunicorn.sh"]
