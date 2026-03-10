from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import search, graph

app = FastAPI(title="AI Research Assistant")

# Allow React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(graph.router)

@app.get("/")
def root():
    return {"status": "running"}