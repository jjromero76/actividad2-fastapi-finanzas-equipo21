# 📈 actividad2-fastapi-finanzas-equipo21

Repository created to build and publish on GitHub an inference API using **FastAPI** for an educational financial use case. The project uses historical market data obtained from **yfinance**, a serialized machine learning model, **Pydantic** data contracts, automated testing, and **Docker** containerization.

---

## 👥 Team Members

- Jhon Jairo Romerosa
- Jackson Cordero Aponte
- William Fernando Blanco

---

# 🚀 Execution Guide

## 1. Open a terminal in the project folder

```powershell
cd C:\[ProjectPath]\actividad2-fastapi-finanzas-equipo21
```

---

## 2. Install Dependencies

Generate and install the project dependencies using Poetry:

```powershell
poetry lock
poetry install
```

---

## 3. Generate Data and Train the Model

Run the following scripts in order:

```powershell
python src/financial_api/data.py
python src/financial_api/features.py
python src/financial_api/train.py
```

These scripts will:

- Download historical market data
- Generate features
- Train the machine learning model
- Save the serialized model artifacts

---

## 4. Run the API with Uvicorn

### PowerShell

```powershell
$env:PYTHONPATH="src"
python -m uvicorn financial_api.api:app --reload --host 127.0.0.1 --port 8000
```

### Open the API Documentation

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 5. Run Automated Tests

Execute all tests with:

```powershell
poetry run pytest
```

---

# 🐳 6. Run with Docker

### Verify Docker Status

```powershell
docker ps
```

This command verifies that Docker is running and displays active containers.

### Build the Docker Image

```powershell
docker build -t actividad2-api .
```

### Run the Container

```powershell
docker run -p 8000:8000 actividad2-api
```

This command starts the FastAPI application inside a Docker container and maps port **8000** from the container to the host machine.

### Access the API Documentation

```text
http://localhost:8000/docs
```

---

## ✅ Project Features

- FastAPI-based REST API
- Historical financial data retrieval
- Machine learning inference
- Pydantic request and response validation
- Automated testing with Pytest
- Docker containerization
- Interactive Swagger documentation