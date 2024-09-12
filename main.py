from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.feature.bot.router import router as chat_router
from src.feature.auth.router import router as auth_router
from src.models.users import User
from src.feature.bot.generator import ask_question, modify_function_parameters

# from passlib.context import CryptContext

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change to specific origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods; adjust if needed
    allow_headers=["*"],  # Allows all headers; adjust if needed
)

routes = APIRouter()
routes.include_router(router=chat_router)
routes.include_router(router=auth_router)
app.include_router(routes,prefix="/api/v1",tags=["Chat"])

# @app.get("/edit-chat-history")
# async def edit_question_from_chat_history(request: Request):
#     return {"message": "hello world"}


# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")




# class ConnectionManager:
#     def __init__(self):
#         self.client_and_connections = {}
    
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()

