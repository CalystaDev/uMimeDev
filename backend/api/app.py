from fastapi import FastAPI
from .routes import router

app = FastAPI()

app.include_router(router)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware( # type: ignore  # noqa  # pylint: disable=no-member
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)