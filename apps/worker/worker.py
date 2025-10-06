import os
import time
from celery import Celery

# Redis connection
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
app = Celery('worker', broker=redis_url, backend=redis_url)

@app.task
def simple_task(message):
    """A simple task that processes a message and returns a result"""
    time.sleep(2)  # Simulate some work
    return f"Processed: {message}"

@app.task  
def add_numbers(x, y):
    """Simple math task"""
    return x + y

if __name__ == '__main__':
    app.start()