from fastapi import APIRouter
from pydantic import BaseModel
import psycopg2
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()

router = APIRouter(prefix="/search", tags=["search"])
model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    source: str | None = None

class SearchResult(BaseModel):
    content: str
    source: str
    page: int
    similarity: float

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
    
class ChatRequest(BaseModel):
    query: str
    source: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query_vector = model.encode(request.query).tolist()
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    if request.source:
        cur.execute("""
            SELECT c.content, s.filename, c.page,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.filename = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5;                    
        """, (query_vector, request.source, query_vector))
    else:
        cur.execute("""
            SELECT c.content, s.filename, c.page,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5;
        """, (query_vector, query_vector))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    context = "\n\n".join([
        f"[Source: {row[1]}, Page{row[2]}]\n{row[0]}"
        for row in rows
    ])
    
    prompt = f"""You are a helpful assistant answering questions about technical manuals.

                Use the following excerpts to answer the question. Be concise and specific.
                If the answer isn't in the excerpts, say so clearly.

                EXCERPTS:
                {context}

                QUESTION: {request.query}

                ANSWER:"""
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt
    )
    
    return ChatResponse(
        answer = response.text,
        sources = [
            SearchResult(
                content=row[0],
                source=row[1],
                page=row[2],
                similarity=round(row[3], 3)
            )
            for row in rows
        ]
    )