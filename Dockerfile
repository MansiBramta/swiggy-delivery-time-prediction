FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY swiggy_delivery_time_prediction/ swiggy_delivery_time_prediction/
COPY models/final_model.joblib models/final_model.joblib

ENV MODEL_SOURCE=local
EXPOSE 8000
CMD ["uvicorn", "swiggy_delivery_time_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]