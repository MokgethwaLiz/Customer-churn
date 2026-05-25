#1. Use the official lightweight Python base image 
FROM python:3.10-slim

#2. set working directory inside the container 
WORKDIR /app

#3. copy only dependency file first (for Docker caching)
COPY requirements.txt .

#4. install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#5. copy the entire project into the image 
COPY . .

#6. Expose FastAPI
EXPOSE 8000

#7. Run FastAPI app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]





