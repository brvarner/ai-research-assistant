from fastapi import APIRouter
from pydantic import BaseModel
import psycopg2
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase
from agents.pipeline import pipeline

load_dotenv()

router = APIRouter(prefix="/search", tags=["search"])
model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
class SearchResult(BaseModel):
    content: str
    source: str
    page: int
    similarity: float
    
class ChatRequest(BaseModel):
    query: str
    source: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    agent_log: list[str] = []

def get_graph_context(query: str, source: str | None) -> str:
    device_map = {
        "CASIO_XWP1.pdf": "CASIO XWP1",
        "EMU_PK6.pdf": "EMU PK6",
        "ENSONIQ_EPS_16_PLUS.pdf": "ENSONIQ EPS 16 PLUS",
        "ENSONIQ_EPS.pdf":"ENSONIQ EPS",
        "KORG_KROME.pdf": "KORG KROME",
        "KORG_M50.pdf": "KORG M50",
        "KORG_NAUTILUS.pdf": "KORG NAUTILUS",
        "KORG_TR.pdf": "KORG TR",
        "KORG_TRITON_LE.pdf": "KORG TRITON LE",
        "KORG_TRITON_PRO.pdf": "KORG TRITON PRO",
        "KORG_TRITON.pdf": "KORG TRITON",
        "ROLAND_FA-06_07_08.pdf": "ROLAND FA",
        "ROLAND_FANTOM-X6_X7_X8.pdf": "ROLAND FANTOM X"
    }
    
    if source and source in device_map:
        devices = [device_map[source]]
    else:
        devices = [name for name in device_map.values()
                    if name.lower() in query.lower()]
    
    graph_sections = []
    
    with neo4j_driver.session() as session:
        for device in devices:
            features = session.run("""
                MATCH(d: Device {name: $device})-[:HAS_SPEC]->(s:Spec)
                RETURN s.name AS name, s.value AS value
            """, device=device).data()
            
            ports = session.run("""
                MATCH (d:Device {name: $device})-[:HAS_PORT]->(p:Port)
                RETURN p.type AS type, p.connector AS connector, 
                       p.direction AS direction
            """, device=device).data()
            
            specs = session.run("""
                MATCH (d:Device {name: $device})-[:HAS_SPEC]->(s:Spec)
                RETURN s.name AS name, s.value AS value
            """, device=device).data()
            
            section = f"Device: {device}\n"
            
            if specs:
                section += "Specs: " + ", ".join(
                    [f"{s['name']}: {s['value']}" for s in specs]
                ) + "\n"
            
            if ports: section += "Ports: " + ", ".join(
                [f"{p['type']} ({p['connector']}, {p['direction']})"
                 for p in ports]
            ) + "\n"
            
            if features:
                section += "Features: " + ", ".join(
                    [f['name'] for f in features]
                ) + "\n"

            graph_sections.append(section)
    return "\n".join(graph_sections)

@router.post("/", response_model=list[SearchResult])
def search(request: SearchRequest):
    query_vector = model.encode(request.query).tolist()

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    if request.source:
        cur.execute("""
            SELECT 
                c.content,
                s.filename,
                c.page,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.filename = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (query_vector, request.source, query_vector, request.limit))
    else:
        cur.execute("""
            SELECT 
                c.content,
                s.filename,
                c.page,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (query_vector, query_vector, request.limit))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        SearchResult(
            content=row[0],
            source=row[1],
            page=row[2],
            similarity=round(row[3], 3)
        )
        for row in rows
    ]
    

    
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = pipeline.invoke({
        "query": request.query,
        "source": request.source,
        "needs_vector": False,
        "needs_graph": False,
        "vector_results": [],
        "graph_context": "",
        "answer": "",
        "agent_log": []
    })
    
    return ChatResponse(
        answer=result["answer"],
        sources=[
            SearchResult(
                content=r["content"],
                source=r["source"],
                page=r["page"],
                similarity=r["similarity"]
            )
            for r in result["vector_results"]
        ],
        agent_log=result.get("agent_log", [])
    )