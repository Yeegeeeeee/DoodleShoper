FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH "${PYTHONPATH}:/app/"

WORKDIR /server

RUN pip3 install poetry==1.7.1

ADD ./db ./db
ADD ./api ./api
ADD ./conf ./conf
ADD ./common ./common
ADD ./models ./models
ADD ./schemas ./schemas
ADD ./services ./services
COPY poetry.lock pyproject.toml main.py __init__.py ./
COPY .env .env

RUN apt-get update && apt-get install -y gcc python3-dev build-essential
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

#gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8888 -w 4 app.main:app