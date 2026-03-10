from fastapi import APIRouter
from pydantic import BaseModel
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/graph", tags=["graph"])

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

class GraphQuery(BaseModel):
    device: str
    
@router.post("/features")
def get_features(query: GraphQuery):
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Device {name: $device})-[:HAS_FEATURE]->(f:Feature)
            RETURN f.name AS feature, f.description AS description
        """, device = query.device)
        
        return[{"feature": row["feature"], "description": row["description"]}
               for row in result]
    
@router.post("/related")
def get_related_devices(query: GraphQuery):
    with driver.session() as session:
        result = session.run("""
            MATCH (d: Device {name: $device})-[:COMPATIBLE_WITH|RELATED_TO]-(other:Device)
            RETURN other.name AS name, other.type AS type
        """, device=query.device)
        return [{"name": row['name'], "type": row["type"]}
                for row in result]