# Deployment

Development commands:

```bash
conda run -n rag python -m pytest -q
conda run -n rag python scripts/run_scholarfusion_smoke.py --pdf-dir data/research_papers --max-papers 3 --max-requests 45 --output-dir evaluation_results/scholarfusion --force
```

Secrets:

- `.env` is ignored.
- API keys are never printed.
- Evaluation reports store IDs, scores and summaries only.

Current async ingestion uses FastAPI BackgroundTasks plus a SQLite task store. It is a single-machine implementation, not a distributed Celery cluster.
