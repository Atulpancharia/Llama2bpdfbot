import asyncio
from fastapi import APIRouter, Request, WebSocket
from pydantic import BaseModel
from datetime import datetime
from src.feature.bot.message import BotMessage
from src.feature.bot.websocket_response import WebSocketResponse
from src.database import get_db
from src.model import initBookProcess
from src.feature.bot.generator import ask_question,modify_function_parameters
import uuid
import json

class ChangeChunkSizeBody(BaseModel):
    new_chunk_size: str
    new_chunk_overlap: str

class QuestionBody(BaseModel):
    query: str

class MessageData(BaseModel):
    chatId: str
    message: str
    isBot: bool
    mt: str

class MessageUploadData(BaseModel):
    mt:str
    message:str
    isBot:bool

class ConnectionManager:
    def __init__(self):
        self.client_and_connections = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()

        if websocket not in self.client_and_connections:
            try:
                # Generate a unique ID for the chat
                new_chat_id = str(uuid.uuid4())
                print(new_chat_id, "new_chat_id")

                self.client_and_connections[websocket] = new_chat_id
                return new_chat_id
            except Exception as e:
                print("Error occurred:", e)
                await websocket.close()
                return None
        else:
            return self.client_and_connections[websocket]
    
    def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.client_and_connections:
                del self.client_and_connections[websocket]
        except Exception as e:
            print("Error in disconnect", e)
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        print("ssssssssssssssssssssssss=================",message)
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print("Error in send_personal_message", e)

manager = ConnectionManager()



router = APIRouter(prefix="/chat")

@router.post("/ask-question")
async def read_root(body: QuestionBody):
    book_process = initBookProcess()

    # Unpack the dictionary into variables
    try:
        model = book_process["model"]
        index = book_process["index"]
        chunks = book_process["chunks"]
        llm = book_process["llm"]
        print("model=========",model,"index===============",index)
    except:
        print("Could not initialize model ! Try changing chunk size or overlap value")
    res=await ask_question(
        query=body.query, model=model, index=index, chunks=chunks, llm=llm
    )
    # print("res==================",res.statusText)
    return res

@router.post("/change-chunk-size")
def read_root(body: ChangeChunkSizeBody):
    return modify_function_parameters(
        "src\model.py", body.new_chunk_size, body.new_chunk_overlap
    )

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    chat_id=await manager.connect(websocket)
    now = datetime.utcnow()
    current_time = now.strftime("%H:%M")
    print(chat_id, "chat_id")
    while True:
        data = await websocket.receive_text()
        print(f"Message from client: {data}")
        await websocket.send_text(f"Message text was: {data}")
        await process_received_data(data, chat_id, current_time, websocket)


async def process_received_data(
    data, chat_id,current_time, websocket: WebSocket
):
    print("process_data=========",data,chat_id)
    received_data = json.loads(data)
    print("received_data=========",received_data)
    mt = received_data.get("mt", "")
    print("mt==========",mt)

    # try:
    #     int(chat_id)
    # except ValueError as e:
    #     print("ValueError", e)


    if mt == "message_upload":
        print("condition function call==============")
        await handle_message_upload(received_data, chat_id, current_time, websocket)


async def handle_message_upload(
    received_data, chat_id, current_time, websocket: WebSocket
):
    print("handle_message_upload===========",received_data)
    upload_data = MessageUploadData(**received_data)
    message = MessageData(
        chatId=chat_id,
        message=upload_data.message,
        isBot=upload_data.isBot,
        mt="message_upload_confirmed",
    )
    print("upload_data.data========",upload_data.message)
    await manager.send_personal_message(websocket, message.dict())
    
    socket_response = WebSocketResponse(
        connection_manager=manager,
        chat_id=chat_id,
        websocket=websocket
    )
    
    bot_message = BotMessage(
        socket_response=socket_response,
        chat_id=chat_id,
        # time_zone=received_data["timezone"],
    )
    no_answer_found = await bot_message.send_bot_message(upload_data.message)
    await asyncio.sleep(1)
    print("no_answer_found=======",no_answer_found)
    
    if no_answer_found==None or no_answer_found==True:
        print("oooooo=================================")
        default_answer = (
            "This is out of context"
        )
        message = MessageData(
            chatId=chat_id,
            message=default_answer,
            isBot=True,
            mt="message_upload_confirmed",
        )
        await manager.send_personal_message(websocket, message.dict())
        
    
