from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="Deepfake API")

# frontend stuff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: change this in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


