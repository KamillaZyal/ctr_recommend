FROM python:3.10-slim

ENV DATA_ROOT /data
ENV PROJECT_ROOT /project_solution

RUN mkdir -p $DATA_ROOT
RUN mkdir -p $PROJECT_ROOT

RUN apt-get update && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "data/download_data.py", ";", "python", "solution/solution.py"]
