from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.logger import logger
from fastapi.exceptions import RequestValidationError
import logging
import asyncio
import cv2
import json
import uuid
from edge_detection_ocr import edge_detection_ocr  # Assuming this function is in another file

# Initialize FastAPI and Logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
task_queue = asyncio.Queue()

def convert_to_ascii(edges):
    ascii_art = ""
    step_size_row = edges.shape[0] // 50
    step_size_col = edges.shape[1] // 10

    for i in range(0, edges.shape[0], step_size_row):
        for j in range(0, edges.shape[1], step_size_col):
            if edges[i, j] > 128:
                ascii_art += "X"
            else:
                ascii_art += " "
        ascii_art += "\n"
    return ascii_art

async def execute_cycle():
    while True:
        try:
            await task_queue.get()
            
            # Capture image from camera
            image_path = await capture_image()
            
            # Perform edge detection and OCR
            edges, ocr_text = await edge_detection_ocr(image_path)
            
            if edges is None or ocr_text is None:
                logger.error("Edge detection or OCR failed.")
                continue
            
            # Convert edges to ASCII art
            ascii_art = convert_to_ascii(edges)
            
            # Save ASCII art to a .json file with a randomized title
            random_title = str(uuid.uuid4()) + ".json"
            with open(random_title, "w") as f:
                json.dump({"ascii_art": ascii_art, "ocr_text": ocr_text}, f)
            
            # Control the vehicle using Llama2
            command = await control_vehicle_with_llama2(edges, ocr_text, ascii_art)
            
            # Send `command` to your vehicle control system
            logger.info(f"Command sent to vehicle: {command}")
            
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
        finally:
            task_queue.task_done()

@app.get("/run_cycle/")
async def run_cycle(background_tasks: BackgroundTasks):
    task_queue.put_nowait("execute")
    background_tasks.add_task(execute_cycle)
    return JSONResponse(content={"status": "Cycle initiated"}, status_code=200)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"An error occurred: {str(exc)}")
    return JSONResponse(content={"message": "Validation error"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

