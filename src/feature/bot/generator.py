import json
import numpy as np
import re
 
from pathlib import Path
from src.model import generate_sample_prompt, initBookProcess, similarity_search
from fastapi.responses import JSONResponse, StreamingResponse
import time
from keybert import KeyBERT
 
async def response_chunk_generator(response: str):
    chunks = response.split()
    for i in chunks:
        time.sleep(0.2)
        yield f"{i} "
 
# Function to clean content by replacing tab characters with single space
def clean_content(content: str) -> str:
    # Replace one or more tab characters with a single space
    return re.sub(r'\t+', ' ', content)

# Function to convert chunks to lowercase and save as JSON
def save_chunks_as_lowercase_json(chunks, file_path):
    lowercase_chunks = [{"chunk_id": idx, "content": chunk.lower()} for idx, chunk in enumerate(chunks)]
    with open(file_path, 'w') as f:
        json.dump(lowercase_chunks, f, indent=2)
    print(f"Chunks saved to {file_path} in lowercase.")

# Define and do the embeddings of query
async def ask_question(query, model, index, chunks, llm):
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
    response = llm.invoke(prompt)
    # response = llm.astream(prompt)
    # final_response = ""
    # print("response++++++++++++++++++++++++++++++", response)
    # async for i in response:
    #     print(i)
    #     final_response = final_response + i
    # print(final_response)
    print("response===========",response)
    
    return StreamingResponse(response_chunk_generator(response), media_type="text/plain")

   
def modify_function_parameters(file_path, new_chunk_size, new_chunk_overlap):
    try:
        # Check if chunk_size is greater than chunk_overlap
        if int(new_chunk_size) < int(new_chunk_overlap):
            return {"Response": "Chunk size must be more than chunk overlap"}
        if int(new_chunk_size) < 11:
            return {"Response": "Chunk size should be more than 10"}
 
        # Define files to delete
        files_to_delete = [
            Path('chunks.json'),
            Path('embeddings.npy'),
            Path('the-story-of-doctor-dolittle.index')
        ]
 
        # Delete the specified files if they exist
        for file in files_to_delete:
            if file.exists():
                file.unlink()
                print(f"File {file} has been removed.")
            else:
                print(f"File {file} does not exist.")
                
        # Read and modify the content of the file
        with open(file_path, 'r') as file:
            content = file.readlines()
            
    
        # Join lines into a single string for easier manipulation
        content_str = ''.join(content)
        print("content_str======",content_str)
        
        # Define patterns for chunk_size and chunk_overlap
        chunk_size_pattern = r'chunk_size=\d+'
        chunk_overlap_pattern = r'chunk_overlap=\d+'
 
        # Replace chunk_size and chunk_overlap values
        new_chunk_size_str = f'chunk_size={new_chunk_size}'
        new_chunk_overlap_str = f'chunk_overlap={new_chunk_overlap}'
 
        content_str = re.sub(chunk_size_pattern, new_chunk_size_str, content_str)
        content_str = re.sub(chunk_overlap_pattern, new_chunk_overlap_str, content_str)
 
        # Split the content string back into lines
        modified_content = content_str.splitlines(keepends=True)
 
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_content)
 
        # Call the initBookProcess function if needed
        # Make sure this function is defined elsewhere
        initBookProcess()
        return {"Response": f"Modified {file_path} with chunk_size={new_chunk_size} and chunk_overlap={new_chunk_overlap}."}
 
    except Exception as e:
        return {"Response": str(e)}

# import json
# import numpy as np
# import re
# from pathlib import Path
# from src.model import generate_sample_prompt, initBookProcess, similarity_search
# from fastapi.responses import JSONResponse, StreamingResponse
# import time
# from keybert import KeyBERT

# async def response_chunk_generator(response: str):
#     chunks = response.split()
#     for i in chunks:
#         time.sleep(0.2)
#         yield f"{i} "

# # Function to clean content by replacing tab characters with single space
# def clean_content(content: str) -> str:
#     return re.sub(r'\t+', ' ', content)

# # Function to convert chunks to lowercase and save as JSON
# def save_chunks_as_lowercase_json(chunks, file_path):
#     lowercase_chunks = [{"chunk_id": idx, "content": chunk.lower()} for idx, chunk in enumerate(chunks)]
#     with open(file_path, 'w') as f:
#         json.dump(lowercase_chunks, f, indent=2)
#     print(f"Chunks saved to {file_path} in lowercase.")

# # Define and do the embeddings of query
# async def ask_question(query, model, index, chunks, pdf_folder, llm):
#     query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()

#     # Perform similarity search using multithreading
#     indices, distances = similarity_search(index, query_embedding)
#     print(f"Indices: {indices}")
#     print(f"Distances: {distances}")
    
#     kw_model = KeyBERT()
#     keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words=None)
    
#     chunksWithSimilarWord = []
#     id = 1
#     found = False
#     print("Keywords: ", keywords)
    
#     for c in chunks:
#         cToDict = c.to_dict()
#         cleaned_content = clean_content(cToDict["page_content"]).lower()
#         for k in keywords:
#             if found:
#                 break
#             if k[0] in cleaned_content.replace("-", "").lower() and len(k[0].split()) > 1:
#                 chunksWithSimilarWord.append({"chunk_id": id, "content": cleaned_content})
#                 id += 1
#                 found = True
#         found = False
    
#     # Limit the chunks to a maximum of 15
#     if len(chunksWithSimilarWord) > 15:
#         chunksWithSimilarWord = chunksWithSimilarWord[:15]
    
#     # Handle cases where indices might be out of range
#     valid_indices = np.array(indices)
#     if valid_indices.size == 0:
#         print("No valid chunks found.")
#         return JSONResponse(content={"Error": "No valid chunks found."}, status_code=404)

#     searched_chunks = [
#         {"chunk_id": int(idx), "content": clean_content(chunks[int(idx)].page_content).lower()}
#         for idx in valid_indices
#         if idx < len(chunks)
#     ]

#     searched_chunks = chunksWithSimilarWord + searched_chunks
    
#     if len(searched_chunks) > 15:
#         searched_chunks = searched_chunks[:15]

#     if not searched_chunks:
#         print("No valid chunks found.")
#         return JSONResponse(content={"Error": "No valid chunks found."}, status_code=404)

#     context_json = json.dumps(searched_chunks, indent=2)
#     prompt = generate_sample_prompt(query, context_json)
    
#     return StreamingResponse(response_chunk_generator(prompt), media_type="text/plain")

