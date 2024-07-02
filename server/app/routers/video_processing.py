from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from queue import Queue, Empty
import threading
import uuid
import os
from app.utils.logger import Logger
from app.algorithms import sit_stand_overall as sso

router = APIRouter()
logger_instance = Logger()
logger = logger_instance.get_logger()

# Dictionary to hold result queues for each UUID
result_queues = {}

def thread_function(video_path, queue, logger):
    try:
        result = sso.sit_stand_overall(video_path, False)
        queue.put(result)
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        queue.put(None)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)  # Ensure the temporary file is deleted

def process_video(file_content: bytes, filename: str, request_id: str, queue: Queue, logger):
    video_path = f"/tmp/{request_id}_{filename}"
    with open(video_path, "wb") as f:
        f.write(file_content)
    thread = threading.Thread(target=thread_function, args=(video_path, queue, logger))
    thread.start()

@router.post("/video_processing")
async def video_processing(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a unique ID for this video processing request
    request_id = str(uuid.uuid4())
    
    # Read the file content into memory
    file_content = await file.read()
    
    # Create a queue for this request ID
    result_queue = Queue()
    result_queues[request_id] = result_queue

    # Start video processing in background
    background_tasks.add_task(process_video, file_content, file.filename, request_id, result_queue, logger)
    logger.info(f"Video processing started in background for request ID {request_id}")
    
    # Respond immediately while processing continues in the background
    return JSONResponse(
        status_code=200,
        content={"message": "Video processing started in background", "request_id": request_id}
    )

@router.get("/video_result/{request_id}")
async def video_result(request_id: str):
    if request_id not in result_queues:
        return JSONResponse(
            status_code=404,
            content={"message": "Invalid request ID"}
        )
    
    result_queue = result_queues[request_id]
    try:
        result = result_queue.get_nowait()  # Retrieve the result from the queue without blocking
    except Empty:
        return JSONResponse(
            status_code=202,
            content={"message": "Video processing is still ongoing"}
        )
    
    if result is None:
        return JSONResponse(
            status_code=500,
            content={"message": "An error occurred during video processing"}
        )
    
    # Clean up the result queue
    del result_queues[request_id]

    type, pass_fail, counter, elapsed_time, rep_durations, violations, max_angles = result
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    logger.info(f"Counter: {counter}")
    logger.info(f"Rep durations: {rep_durations}")
    logger.info(f"Violations: {violations}")
    logger.info(f"Max angles: {max_angles}")
    
    return JSONResponse(
        status_code=200,
        content={
            "type": type,
            "pass_fail": pass_fail,  # "pass" or "fail
            "counter": counter,
            "elapsed_time": elapsed_time,
            "rep_durations": rep_durations,
            "violations": violations,
            "max_angles": max_angles
        }
    )
