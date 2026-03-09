from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import search

app = FastAPI(title="AI Research Assistant")

# Allow React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)

@app.get("/")
def root():
    return {"status": "running"}