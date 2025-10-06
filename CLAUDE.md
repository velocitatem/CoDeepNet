# Template
My template repository for almost any project that I might be doing in the field of AI or building a software product or any sort of tool ranging form a machine learning project, data analysis, SaaS app or a python library or some sort of API or scraper or ETL tool or some sort of simulation or just a small python program to run something simple.
This template is AI native and platform agnostic and meant to be effortlessly deployable to anywhere at any time with any software with minimal effort managed via my Makefile.

#### Directory Breakdown
- apps
  - webapp (A next.js 15.5.2 webapp with react 19 pre-configured with basics and Tailwind CSS)
    - This can server both as the frontend but also nextjs allows for API route definitions for simple things that we could build.
  - webapp-minimal (Streamlit webapp with minimal web interface for quick prototypes)
  - worker (Background worker template for long running tasks for programs)
  - backend/{fastapi|flask} to define a web api with either or whichever is specified/user wants or is a best fit scenario.
- ml (Machine learning pipeline for PyTorch)
  - models (arch.py where we define architectures and train.py for the training loop)
  - inference.py (A minimal setup webserver with fastapi to run inference online)
  - notebooks (For any notebook needs)
  - data (has etl.py for any ETL and should be a single place for turning raw data into pytorch ready datasets)
- src (Just as __init__.py for any simple modules or building libraries within there, should be used for simple python scripts without any otehr needs or just running something in the CLI)


#### Services Associated
- I setup a basic optional minio service to run for any needs of object storage or manipulation for any machine learning tasks
- Tensorboard is very useful for monitoring experiments and is defined as a docker service an spinupable with the make tensorboard command.

###### Logging
1. Grafana to view (must be configured by adding loki url with "http://loki:31000")
2. For now just python directly adds the logs to loki (via the alveslib package)

```python
from alveslib import get_logger
logger = get_logger("service")
```
FOR REFERENCE ALL OTHER REUSABLE MODULES LIKE THIS SHOULD BE DEFINED THE SAME WAY IN THE ALVESLIB package - if used in python.
Using lazydocker to manager containers...

### Checklists and Best Practices and Code Hygine

#### `apps/webapp` - building a nextjs app.

Reusability is key - define modules and do not repeat code or logic anywhere. Style should be done LAST, when defining new components define them bare-bones or motivated with globals.css if compelted.

- [ ] Using a provided moodboard -> create a globals.css update to match the style [use creative structure for each project differently (do not take provided as given)]
- [ ] Flesh out proper content for the TOC and Privacy Policy - write content for both of the components to properly define them
- [ ] Optionally turn off eslint
- [ ] Properly stylize the Header and Footer
- [ ] Define the robots.txt and llms.txt
- [ ] Connect a supabase project if it is being used and properly spin up a local supabase container setup if desired for local testing. If a webapp does not require user auth (or just yet) just remove references to the /login /dashboard routes but do not delete the code of using them.
- [ ] Connect analytics with google with `gtag package`

#### `ml/data` - using and building datasets for ML
Parallel data loading and using third party datasets. All code written must ensure that running the model training is possible on any machine.
Downloading data from third party sources must be done in a reproducible way (the export part of the ETL). If datasets are too large for just pandas, using spark is the best way forward.
Data processing should be cachable so if any stage of the transformation or data processing fails it does not start from scrach. If a dataloader class is being used and data is not present on the system, it should trigger logic to get the data and handle any transformations necssary. The whole pipeline should be self informed about what is hapenning and aware of the phase its in. If third party requests are made ferquently, they should also be cached to prevent overloading any servers.
Dataloaders defined in pytorch if handling for example 1e6 images with a storage bucket like minio (just a dummy example it should be generaliziable) must stream the data as it is being used for training just in time.

#### `ml/models` - creating model architectures and defining training loops
Define the architectuer in arch.py and training loop in train.py - training should be logged with tensorboard always and evaluations metrics should be versioned and defined in separate logic units like eval.py to make experiments comparable, if at any poitns eval metrics change or scale they should be tracked under a sparate track of experiments in tensorboard - do not arbitrarily modify the eval of anything or loss unless explicitly necessary.
Experiments should be tracked with meaningful names and logging any parameters into experiments. Machine learning is a life cycle and always evolving. DO NOT create any rogue execution scripts just use the train.py for training.
