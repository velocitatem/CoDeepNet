from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    PORT=int(os.getenv("BACKEND_PORT", 5000))
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
