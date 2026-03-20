from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import psycopg2
import operator
import os
from dotenv import load_dotenv

load_dotenv()

class ResearchState(TypedDict):
    query: str
    source: str | None
    needs_vector: bool 
    needs_graph: bool 
    vector_results: list
    graph_context: str
    answer: str
    agent_log: Annotated[list[str], operator.add]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

DEVICE_MAP = {
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

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )

def router_agent(state: ResearchState) -> ResearchState:
    """Decides which agents to invoke based on the query."""
    query = state["query"].lower()

    graph_keywords = [
        "port", "connect", "midi", "usb", "spec", "polyphony",
        "feature", "has", "support", "interface", "jack", "output",
        "input", "compare", "difference", "versus", "vs", "sd card", 
        "storage", "memory", "sampling"
    ]
    vector_keywords = [
        "how", "what", "explain", "guide", "setup", "use",
        "configure", "program", "edit", "create", "play", "record",
        "why", "when", "where", "step"
    ]

    needs_graph = any(k in query for k in graph_keywords)
    needs_vector = any(k in query for k in vector_keywords)

    # Default to both if unclear
    if not needs_graph and not needs_vector:
        needs_graph = True
        needs_vector = True

    log = state.get("agent_log", [])
    log.append(f"Router: needs_vector={needs_vector}, needs_graph={needs_graph}")

    return {
        **state,
        "needs_vector": needs_vector,
        "needs_graph": needs_graph,
        "agent_log": [f"Router: needs_vector={needs_vector}, needs_graph={needs_graph}"]
    }
    
def vector_agent(state: ResearchState) -> dict:
    if not state.get("needs_vector"):
        return {
            "vector_results": [],
            "agent_log": ["Vector Agent: skipped"]
        }

    query_vector = embedding_model.encode(state["query"]).tolist()
    conn = get_db_conn()
    cur = conn.cursor()

    if state.get("source"):
        cur.execute("""
            SELECT c.content, s.filename, c.page,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.filename = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5;
        """, (query_vector, state["source"], query_vector))
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

    results = [
        {"content": r[0], "source": r[1], "page": r[2], "similarity": round(r[3], 3)}
        for r in rows
    ]

    return {
        "vector_results": results,
        "agent_log": [f"Vector Agent: found {len(results)} chunks"]
    }


def graph_agent(state: ResearchState) -> dict:
    if not state.get("needs_graph"):
        return {
            "graph_context": "",
            "agent_log": ["Graph Agent: skipped"]
        }

    if state.get("source") and state["source"] in DEVICE_MAP:
        devices = [DEVICE_MAP[state["source"]]]
    else:
        devices = [
            name for name in DEVICE_MAP.values()
            if name.lower() in state["query"].lower()
        ]
        if not devices:
            devices = list(DEVICE_MAP.values())

    sections = []

    with neo4j_driver.session() as session:
        for device in devices:
            features = session.run("""
                MATCH (d:Device {name: $device})-[:HAS_FEATURE]->(f:Feature)
                RETURN f.name AS name
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

            if not features and not ports and not specs:
                continue

            section = f"{device}:\n"
            if specs:
                section += "  Specs: " + ", ".join(
                    [f"{s['name']}: {s['value']}" for s in specs]
                ) + "\n"
            if ports:
                section += "  Ports: " + ", ".join(
                    [f"{p['type']} ({p['connector']})" for p in ports]
                ) + "\n"
            if features:
                section += "  Features: " + ", ".join(
                    [f['name'] for f in features]
                ) + "\n"

            sections.append(section)

    graph_context = "\n".join(sections)
    return {
        "graph_context": graph_context,
        "agent_log": [f"Graph Agent: retrieved data for {len(sections)} devices"]
    }


def synthesis_agent(state: ResearchState) -> dict:
    vector_context = "\n\n".join([
        f"[Source: {r['source']}, Page {r['page']}]\n{r['content']}"
        for r in state.get("vector_results", [])
    ])

    graph_context = state.get("graph_context", "")

    prompt = f"""You are a helpful assistant answering questions about
keyboard and synthesizer manuals.

You have access to two sources of information:

1. STRUCTURED DEVICE DATA (knowledge graph):
{graph_context if graph_context else "Not queried for this request."}

2. MANUAL EXCERPTS (semantic search):
{vector_context if vector_context else "Not queried for this request."}

Use both sources to give the most complete and accurate answer possible.
Be specific — reference specs, port names, and features by name when relevant.
If the answer isn't available in either source, say so clearly.

QUESTION: {state["query"]}

ANSWER:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "agent_log": ["Synthesis Agent: answer generated"]
    }


# ─── Routing logic ────────────────────────────────────────────────────────────
def route_after_router(state: ResearchState) -> str:
    """After routing, always run both agents in sequence."""
    return "vector_agent"


# ─── Build graph ──────────────────────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(ResearchState)

    graph.add_node("router_agent", router_agent)
    graph.add_node("vector_agent", vector_agent)
    graph.add_node("graph_agent", graph_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    graph.set_entry_point("router_agent")
    graph.add_edge("router_agent", "vector_agent")
    graph.add_edge("vector_agent", "graph_agent")
    graph.add_edge("graph_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph.compile()

pipeline = build_pipeline()