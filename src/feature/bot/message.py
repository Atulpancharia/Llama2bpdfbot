import asyncio
import json
import numpy as np
from src.feature.bot.generator import ask_question, response_chunk_generator, save_chunks_as_lowercase_json
from src.model import clean_content, generate_sample_prompt, initBookProcess, similarity_search
from src.feature.bot.utilities.greetings_response import create_greetings_response
from src.feature.bot.utilities.local_semantic_similarity.auto_reply import can_auto_reply
from .websocket_response import WebSocketResponse
from fastapi.responses import JSONResponse, StreamingResponse
from keybert import KeyBERT


class BotMessage:
    def __init__(
        self,
        *,
        chat_id: str,
        socket_response: WebSocketResponse,
    ):
        self.chat_id = chat_id
        self.socket_response = socket_response
        
    async def auto_reply_handler(
        self, last_user_text: str, auto_reply_type: str
    ) -> bool:
        """
        Handles if bot can reply directly to the user
        """
        if can_auto_reply(last_user_text):
            msg_type = None
            msg = create_greetings_response()
            print("msg=====>", msg)
            await self.socket_response.create_bot_response(
                msg,
                None,
                msg_type="greetings_msg"
            )
            return True
        return False
        
    
    async def send_bot_message(self, text: str) -> bool:
        """
        Sends a bot message to the user
        :param text: the text the user sent
        :returns: whether the bot has sent the response or not
        """
        # Task for handling the main bot logic
        is_answer_found_and_sent_main_task =  await asyncio.create_task(self.bot_handler(text))
        return is_answer_found_and_sent_main_task
    
    async def bot_handler(self, user_input: str) -> bool:
        if await self.auto_reply_handler(user_input, "greeting"):
            return False
        
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
        res=await self.ask_question(
            query=user_input, model=model, index=index, chunks=chunks, llm=llm
        )
        print("res=================",res)
        if(res):
            print("resssssss===============")
            return False
        else:
            return True
        
        # print("res==================",res.statusText)
        # return res
        
    
    async def ask_question(self,query, model, index, chunks, llm):
        query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    
        # Convert chunks to lowercase and save to JSON
        save_chunks_as_lowercase_json([chunk.page_content for chunk in chunks], 'chunks_lowercase.json')
    
        # Perform similarity search
        indices, distances = similarity_search(index, query_embedding)
        print(f"Indices: {indices}")
        print(f"Distances: {distances}")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words=None)
        chunksWithSimilarWord = []
        id = 1
        found = False
        print("Keywords: ", keywords)
        for c in chunks:
            cToDict = c.to_dict()
            # Clean the content before processing
            cleaned_content = clean_content(cToDict["page_content"]).lower()  # Convert content to lowercase
            for k in keywords:
                if found:
                    break
                if k[0] in cleaned_content.replace("-", "").lower() and len(k[0].split()) > 1:
                    chunksWithSimilarWord.append({"chunk_id": id, "content": cleaned_content})
                    id += 1
                    found = True
            found = False
        
        # Limit the chunks to a maximum of 15
        if len(chunksWithSimilarWord) > 8:
            chunksWithSimilarWord = chunksWithSimilarWord[:8]
        
        # Handle cases where indices might be out of range
        valid_indices = np.array(indices)
        if valid_indices.size == 0:
            print("No valid chunks found.")
            return JSONResponse(content={"Error": "No valid chunks found."}, status_code=404)
    
        searched_chunks = [
            {"chunk_id": int(idx), "content": clean_content(chunks[int(idx)].page_content).lower()}
            for idx in valid_indices
            if idx < len(chunks)
        ]
    
        searched_chunks = chunksWithSimilarWord + searched_chunks
    
        # Again limit the total number of chunks to 15, in case combining chunksWithSimilarWord and searched_chunks exceeds 15
        if len(searched_chunks) > 15:
            searched_chunks = searched_chunks[:15] + chunksWithSimilarWord
    
        if not searched_chunks:
            print("No valid chunks found.")
            return JSONResponse(content={"Error": "No valid chunks found."}, status_code=404)
    
        # Convert searched chunks to JSON format
        context_json = json.dumps(searched_chunks, indent=2)
        # Generate prompt
        print("context_json===========================================================", context_json)
        prompt = generate_sample_prompt(query, context_json)
        # Get the full response
        # response = llm.invoke(prompt)
        response = llm.astream(prompt)
        final_response = ""
        print("response++++++++++++++++++++++++++++++", response)
        
        await self.socket_response.create_bot_response(
                    response,
                    None,
                    msg_type="normal_msg"
                )
        # async for i in response:
        #     print(i)
            
        #     final_response = final_response + i
        # # print(final_response)
        # print("response===========",final_response)
        
        # async def string_to_generator(data):
        #     for char in data:
        #         yield char
        #         await asyncio.sleep(0.01)
                
        # generator = string_to_generator(response)
        # print("generator=======",generator)
        # generator=StreamingResponse(response_chunk_generator(response), media_type="text/plain")
        # print("generator==========",generator)
        await self.socket_response.create_bot_response(
                    final_response,
                    None,
                    msg_type="normal_msg"
                )
        return response
        
        


