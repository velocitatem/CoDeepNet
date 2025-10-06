# Template
My template repository for almost any project that I might be doing in the field of AI or building a software product or any sort of tool ranging form a machine learning project, data analysis, SaaS app or a python library or some sort of API or scraper or ETL tool or some sort of simulation or just a small python program to run something simple.
This template is AI native and platform agnostic and meant to be effortlessly deployable to anywhere at any time with any software with minimal effort managed via my Makefile.

#### Directory Breakdown
- apps
  - webapp (A next.js 15.5.2 webapp with react 19 pre-configured with basics and Tailwind CSS)
    - This can server both as the frontend but also nextjs allows for API route definitions for simple things that we could build.
  - webapp-minimal (Streamlit webapp with minimal web interface for quick prototypes)
  - worker (Background worker template for long running tasks for programs)
- ml (Machine learning pipeline for PyTorch)
  - models (arch.py where we define architectures and train.py for the training loop)
  - inference.py (A minimal setup webserver with fastapi to run inference online)
  - notebooks (For any notebook needs)
  - data (has etl.py for any ETL and should be a single place for turning raw data into pytorch ready datasets)
- src (Just as __init__.py for any simple modules or building libraries within there, should be used for simple python scripts without any otehr needs or just running something in the CLI)



#### Services Associated
- I setup a basic optional minio service t orun for any needs of object storage or manipulation for any machine learning tasks
- Tensorboard is very useful for monitoring experiments and is defined as a docker service an spinupable with the make tensorboard command.

###### Logging
1. Grafana to view (must be configured by adding loki url with "http://loki:31000")
2. For now just python directly adds the logs to loki (via the alveslib package)

```python
from alveslib import get_logger
logger = get_logger("service")
```

FOR REFERENCE ALL OTHER REUSABLE MODULES LIKE THIS SHOULD BE DEFINED THE SAME WAY IN THE ALVESLIB package
Using lazydocker to manager containers...
