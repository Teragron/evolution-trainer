FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY main.py .
COPY evolution ./evolution
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["./entrypoint.sh"]
