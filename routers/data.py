import os
import random
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.database import get_db
from database.models import User, PredictionLog
from config import config

router = APIRouter(tags=["data"])

# Helper to load data from main memory if possible
def _get_dataset_data(dataset: str):
    from main import load_dataset_resources, DATASET_CFG
    if dataset not in DATASET_CFG:
        raise ValueError("Invalid dataset")
    return load_dataset_resources(dataset)


@router.get("/drugs")
async def get_drugs(
    dataset: str = "C-dataset", 
    page: int = 1, 
    limit: int = 20, 
    search: str = "", 
    sort_by: str = "name", 
    order: str = "asc"
):
    try:
        if dataset == "C": dataset = "C-dataset"
        elif dataset == "B": dataset = "B-dataset"
        elif dataset == "F": dataset = "F-dataset"

        model, drug_sim, disease_sim, d_names, d_smiles, di_names, node_ids = _get_dataset_data(dataset)
        
        # Build mock data for now since we don't have the full adjacency loaded in memory
        results = []
        for i, name in enumerate(d_names):
            if search.lower() in name.lower():
                results.append({
                    "id": node_ids[i],
                    "name": name,
                    "dataset": dataset[0],
                    "degree": random.randint(1, 20),
                    "top_diseases": random.sample(di_names, min(3, len(di_names)))
                })
        
        if sort_by == "name":
            results.sort(key=lambda x: x["name"], reverse=(order == "desc"))
        elif sort_by == "degree":
            results.sort(key=lambda x: x["degree"], reverse=(order == "desc"))
            
        total = len(results)
        start = (page - 1) * limit
        end = start + limit
        
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "data": results[start:end]
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diseases")
async def get_diseases(
    dataset: str = "C-dataset", 
    page: int = 1, 
    limit: int = 20, 
    search: str = "", 
    sort_by: str = "name", 
    order: str = "asc"
):
    try:
        if dataset == "C": dataset = "C-dataset"
        elif dataset == "B": dataset = "B-dataset"
        elif dataset == "F": dataset = "F-dataset"

        model, drug_sim, disease_sim, d_names, d_smiles, di_names, node_ids = _get_dataset_data(dataset)
        
        num_drugs = len(d_names)
        results = []
        for i, name in enumerate(di_names):
            if search.lower() in name.lower():
                results.append({
                    "omim_id": node_ids[num_drugs + i] if (num_drugs + i) < len(node_ids) else f"OMIM:{10000+i}",
                    "id": node_ids[num_drugs + i] if (num_drugs + i) < len(node_ids) else f"OMIM:{10000+i}",
                    "name": name,
                    "dataset": dataset[0],
                    "degree": random.randint(1, 20),
                    "top_drugs": random.sample(d_names, min(3, len(d_names)))
                })
                
        if sort_by == "name":
            results.sort(key=lambda x: x["name"], reverse=(order == "desc"))
        elif sort_by == "degree":
            results.sort(key=lambda x: x["degree"], reverse=(order == "desc"))
            
        total = len(results)
        start = (page - 1) * limit
        end = start + limit
        
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "data": results[start:end]
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/proteins")
async def get_proteins(
    dataset: str = "C-dataset", 
    page: int = 1, 
    limit: int = 20, 
    search: str = ""
):
    try:
        # Mock proteins since there is no direct protein file loaded by default
        # Create a stable list of proteins
        random.seed(42)
        all_proteins = [{"id": f"P{i:04d}", "name": f"Protein_{i}", "related_drugs": random.randint(0, 10), "related_diseases": random.randint(0, 10)} for i in range(1, 4756)]
        
        results = [p for p in all_proteins if search.lower() in p["name"].lower()]
        
        total = len(results)
        start = (page - 1) * limit
        end = start + limit
        
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "data": results[start:end]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/network")
async def get_graph_network(
    dataset: str = "C-dataset", 
    drug_limit: int = 30, 
    disease_limit: int = 60, 
    show_protein: str = "false",
    search: str = ""
):
    try:
        if dataset == "C": dataset = "C-dataset"
        elif dataset == "B": dataset = "B-dataset"
        elif dataset == "F": dataset = "F-dataset"

        model, drug_sim, disease_sim, d_names, d_smiles, di_names, node_ids = _get_dataset_data(dataset)
        
        show_prot = show_protein.lower() == 'true'
        
        sel_drugs = d_names[:drug_limit]
        sel_diseases = di_names[:disease_limit]
        
        nodes = []
        for i, d in enumerate(sel_drugs):
            nodes.append({"id": f"drug_{i}", "label": d, "type": "drug", "group": "drug", "x": random.random(), "y": random.random()})
            
        for i, di in enumerate(sel_diseases):
            nodes.append({"id": f"dis_{i}", "label": di, "type": "disease", "group": "disease", "x": random.random(), "y": random.random()})
            
        edges = []
        for i in range(len(sel_drugs)):
            for _ in range(random.randint(1, 3)):
                edges.append({
                    "source": f"drug_{i}",
                    "target": f"dis_{random.randint(0, len(sel_diseases)-1)}",
                    "weight": random.random()
                })
                
        if show_prot:
            for i in range(15):
                nodes.append({"id": f"prot_{i}", "label": f"Protein_{i}", "type": "protein", "group": "protein", "x": random.random(), "y": random.random()})
                edges.append({
                    "source": f"prot_{i}",
                    "target": f"drug_{random.randint(0, len(sel_drugs)-1)}",
                    "weight": random.random()
                })
                
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "drug_count": len(sel_drugs),
                "disease_count": len(sel_diseases)
            }
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
