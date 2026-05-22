FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

COPY blip_finetuned /app/blip_finetuned

EXPOSE 5000

CMD ["python", "app.py"]