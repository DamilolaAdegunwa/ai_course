from fastapi import FastAPI, HTTPException
from typing import List  # , Dict
from pydantic import BaseModel
app = FastAPI()

# In-memory storage of itineraries
itineraries = {}

# request
class Destination(BaseModel):
    city: str
    country: str
    days: int

class ItineraryRequest(BaseModel):
    user_id: str
    destinations: List[Destination]

class Itinerary:
    def __init__(self, user_id: str, destinations: List[Destination]):
        self.user_id = user_id
        self.destinations = destinations

"""
@app.post("/create_itinerary/")
def create_itinerary(user_id: str, destinations: List[Dict]):
    print(f"{user_id}: str, {destinations}: List[Dict]")
    if user_id in itineraries:
        raise HTTPException(status_code=400, detail="Itinerary already exists for user.")
    itinerary = Itinerary(user_id, destinations)
    itineraries[user_id] = itinerary
    return {"message": "Itinerary created", "itinerary": destinations}
"""


@app.post("/create_itinerary/")
def create_itinerary(request: ItineraryRequest):
    if request.user_id in itineraries:
        raise HTTPException(status_code=400, detail="Itinerary already exists for user.")

    itinerary = Itinerary(user_id=request.user_id, destinations=request.destinations)
    itineraries[request.user_id] = itinerary

    return {"message": "Itinerary created", "itinerary": request.destinations}

@app.get("/get_itinerary/{user_id}")
def get_itinerary(user_id: str):
    if user_id not in itineraries:
        raise HTTPException(status_code=404, detail="Itinerary not found.")
    return itineraries[user_id].destinations

# run the following: uvicorn itinerary_service:app --reload --port 8000
# http://localhost:8000/docs for the swagger