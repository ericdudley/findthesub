FROM python:3.13.1-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8008
CMD ["gunicorn", "--bind", "0.0.0.0:8008", "findthesub.wsgi:application"]
