FROM python:3.11-slim

WORKDIR /app

COPY maze_runner_rl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY maze_runner_rl/ .

EXPOSE 8765

CMD ["python", "main.py"]