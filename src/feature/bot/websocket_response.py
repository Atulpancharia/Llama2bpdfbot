from fastapi import WebSocket
from datetime import datetime
import json
import re
from uuid import uuid4
from fastapi import logger
from typing import Optional
from collections.abc import AsyncGenerator

class WebSocketResponse:
    def __init__(
        self,
        *,
        chat_id: str,
        websocket: WebSocket,
        connection_manager=None,
    ):
        self.chat_id = chat_id
        self.connection_manager = connection_manager
        self.websocket = websocket
    
    async def async_word_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Async generator yielding words from the given text."""
        words = re.split(r"(\s+)", text)
        for word in words:
            yield word

    async def create_bot_response(
        self,
        text: AsyncGenerator[str, None] | str,
        time_zone: Optional[str] = None,
        user_question: Optional[str] = None,
        *,
        msg_type: Optional[str] = "normal_msg",
    ) -> Optional[str]:
        """Creates bot message and sends it. Returns full text."""
        final_text = ""
        print("text===>", text, "typee===============", type(text))
        # text = self.async_word_generator(text)
        
        if isinstance(text, str):
            text = self.async_word_generator(text)
            
        if True:
            async for t in text:
                if t:
                    final_text += t
                    # message = MessagePartialUpload(
                    #     mt="chat_message_bot_partial", sid=self.sid, partial=t
                    # )
                    await self.connection_manager.send_personal_message(
                        self.websocket,
                        {
                            "mt":("chat_message_bot_partial"),
                            "chatId": self.chat_id,
                            "partial":t
                        }
                    )

            # message = MessageData(
            #     time=current_time,
            #     sid=self.sid,
            #     message=text,
            #     isBot=True,
            #     mt="message_upload_confirm",
            # )

            # await self.sio.emit("new_message", json.dumps(message.dict()))
            await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "message_upload_confirmed",
                "chatId": self.chat_id,
                "message": final_text,
                "isBot": True,
            },
        )
    
