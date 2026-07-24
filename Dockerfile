FROM python:3.13-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-root

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "financial_api.api:app", "--host", "0.0.0.0", "--port", "8000"]