from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.logger import logger
from fastapi.exceptions import RequestValidationError
import logging
import asyncio
from edge_detection_ocr import edge_detection_ocr
from stable_diffusion import simulate_pathways
from camera_input import capture_image
from llama2_control import control_vehicle_with_llama2

# Initialize FastAPI and Logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
task_queue = asyncio.Queue()

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
            
            # Simulate pathways using stable diffusion
            simulated_pathways = await simulate_pathways(edges)
            
            # Control the vehicle using Llama2
            command = await control_vehicle_with_llama2(edges, ocr_text, simulated_pathways)
            
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
