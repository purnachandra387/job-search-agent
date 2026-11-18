FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY load_jobs.py search.py agent.py app.py eval_search.py sample_jobs.csv ./
COPY test_search.py test_app.py test_agent.py test_eval.py ./
RUN python load_jobs.py sample_jobs.csv && python search.py build
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
