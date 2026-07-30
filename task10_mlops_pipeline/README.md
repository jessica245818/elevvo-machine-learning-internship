# Task 10: End-to-End MLOps Pipeline

This project serves the trained Task 9 predictive-maintenance model as a
validated FastAPI service and packages it as a reproducible Docker container.
A Streamlit interface and GitHub Actions CI workflow complete the deployment
pipeline.

## Architecture

```text
JSON / CSV client
       |
       v
Pydantic schema validation --invalid--> HTTP 400
       |
       v
Feature engineering
       |
       v
Task 9 Random Forest + threshold 0.81
       |
       v
Failure probability and maintenance decision
```

The 4.3 MB model bundle is copied from Task 9 and committed with this service,
so deployments do not need the training data or a training step.

## API contract

`POST /predict` accepts:

```json
{
  "product_type": "M",
  "air_temperature_k": 300.1,
  "process_temperature_k": 310.2,
  "rotational_speed_rpm": 1500.0,
  "torque_nm": 40.0,
  "tool_wear_min": 100.0
}
```

Numerical fields use strict numeric validation and physical range limits.
Product type must be `L`, `M`, or `H`; unknown fields are forbidden. Invalid
types, malformed JSON, out-of-range values, empty batches, and unexpected
fields return `400 Bad Request` with structured error details.

Endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Container and model readiness |
| `POST` | `/predict` | One validated sensor reading |
| `POST` | `/predict/batch` | 1–1,000 validated readings |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Run locally

```bash
python -m pip install -r task10_mlops_pipeline/requirements-api.txt
uvicorn task10_mlops_pipeline.app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @task10_mlops_pipeline/sample_request.json
```

## Run with Docker

From the Task 10 directory:

```bash
docker build -t predictive-maintenance-api .
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Run the API and frontend together:

```bash
docker compose up --build
```

- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit frontend: [http://localhost:8501](http://localhost:8501)

## Test

```bash
python -m pip install -r task10_mlops_pipeline/requirements-dev.txt
python -m pytest task10_mlops_pipeline/tests -q
```

The tests cover health/readiness, successful inference, response schemas,
malformed JSON, text in numeric fields, invalid product categories,
out-of-range values, unknown fields, and empty batches.

`postman_collection.json` can be imported directly into Postman. It includes
health, valid prediction, and intentionally invalid `400` validation requests.

## CI/CD bonus

`.github/workflows/task10-ci.yml` runs the API test suite and builds the Docker
image on every push or pull request that changes Task 10. The workflow has
read-only repository permissions and uses pinned major versions of official
GitHub actions.

## Frontend bonus

The Streamlit application supports:

- Interactive entry of one sensor reading
- CSV uploads for batch predictions
- Failure probability and maintenance decisions
- Clear API error reporting
