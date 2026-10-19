"""Serve the final model: send one client's details, get back whether to call them.

The API loads the bundle saved by `python -m src.final` (fitted Pipeline + frozen thresholds), so
it receives raw client fields and applies exactly the preprocessing used in training.
`duration` is deliberately absent: it is only known after the call.

Start:  uvicorn src.api:app --reload      then open http://127.0.0.1:8000/docs to try it
"""
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CONFIG, path
from src.fairness import age_group

COST = CONFIG["business"]["cost_per_call"]
REVENUE = CONFIG["business"]["revenue_per_subscription"]
MODEL_FILE = path("models") / "final_model.joblib"
state: dict = {}


class Client(BaseModel):
    """One client, as known BEFORE the call. Categories are the ones seen in training."""
    age: int = Field(ge=18, le=100)
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                 "self-employed", "services", "student", "technician", "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["no", "yes"] = Field(description="Has credit in default?")
    balance: int = Field(description="Average yearly balance in EUR (can be negative)")
    housing: Literal["no", "yes"] = Field(description="Has a housing loan?")
    loan: Literal["no", "yes"] = Field(description="Has a personal loan?")
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(ge=1, le=31, description="Day of month of the planned call")
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    campaign: int = Field(ge=1, description="Calls in this campaign, including this one")
    pdays: int = Field(ge=-1, description="Days since last contact in a previous campaign (-1 = never)")
    previous: int = Field(ge=0, description="Contacts before this campaign")
    poutcome: Literal["failure", "other", "success", "unknown"]

    model_config = {"json_schema_extra": {"examples": [{
        "age": 36, "job": "technician", "marital": "divorced", "education": "secondary",
        "default": "no", "balance": 861, "housing": "no", "loan": "no", "contact": "cellular",
        "day": 29, "month": "aug", "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown",
    }]}}


class Decision(BaseModel):
    probability: float = Field(description="Estimated chance the client subscribes")
    expected_value_eur: float = Field(description="probability x revenue - call cost")
    call: bool = Field(description="Business rule: one profit-optimal threshold for everyone")
    call_age_fair: bool = Field(description="Fairness rule: age-group thresholds (equal opportunity)")
    threshold: float
    age_group: str
    age_group_threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at start-up, not on every request.
    if not MODEL_FILE.exists():
        raise RuntimeError(f"{MODEL_FILE} not found: run `python -m src.final` first.")
    state.update(joblib.load(MODEL_FILE))
    yield
    state.clear()


app = FastAPI(title="Bank Marketing: should we call this client?", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Decision)
def predict(client: Client) -> Decision:
    row = pd.DataFrame([client.model_dump()])[state["columns"]]
    try:
        p = float(state["model"].predict_proba(row)[0, 1])
    except Exception as error:  # a malformed row should be a clear 422, not a server crash
        raise HTTPException(status_code=422, detail=str(error)) from error
    group = str(age_group(row["age"])[0])
    group_threshold = state["group_thresholds"][group]
    return Decision(
        probability=round(p, 4), expected_value_eur=round(p * REVENUE - COST, 2),
        call=p >= state["threshold"], call_age_fair=p >= group_threshold,
        threshold=state["threshold"], age_group=group, age_group_threshold=group_threshold,
    )
