"""Local HTTP integration for A1 Activity Generation and analytics."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from activity_data import activity_text
from predict_example import ActivityClassifier


class Activity(BaseModel):
    title: str = Field(default="", max_length=2000)
    description: str = Field(default="", max_length=20000)
    instructions: str = Field(default="", max_length=20000)

    @model_validator(mode="after")
    def validate_text(self):
        activity_text(self.model_dump())
        return self


@asynccontextmanager
async def lifespan(app):
    app.state.classifier = ActivityClassifier(os.getenv("MODEL_PATH"))
    yield


app = FastAPI(title="MyVillage Activity Type Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/predict")
def predict(activity: Activity):
    return app.state.classifier.predict(activity.model_dump())
