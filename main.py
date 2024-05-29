from fastapi import FastAPI
from routes import predict, result
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Air Gap Detection API",
              description="API for detecting air gaps in images", version="0.1.0")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(result.router)
