from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from typing import Annotated
import logging
from conf.config import Config

from db.db import get_db, AsyncIOMotorClient
from schemas.user_resource import (
    add_thread_user_resource as db_add_thread_user_resource,
    remove_thread_user_resource as db_remove_thread_user_resource,
    get_user_resource as db_get_user_resource,
)

from models.threads import (
    CreateThreadResourceResp,
    ListThreadResourceResp,
    SendMessageResourceReq,
    ListMessageResourceResp,
    SendMessageResourceResp,
    RunThreadResp,
    RunThreadStatusResp,
    SubmitToolsReqText
)

from services.google_search import fetch_search_results_text

from common.util import get_current_user
from common.constants import *

from openai import OpenAI

from services.mediapipe_recognition.main import sign_language_recognition

router = APIRouter()

client = OpenAI(api_key=Config.app_settings.get('openai_key'), default_headers={"OpenAI-Beta": "assistants=v2"})


@router.post('/sign', status_code=status.HTTP_200_OK)
async def sign_thread_new(
        current_user: Annotated[str, Depends(get_current_user)],
        db: AsyncIOMotorClient = Depends(get_db),  # type: ignore
):
    logging.info(f'Creating new thread for: %s', Config.app_settings.get('openai_assistant'))

    new_thread = client.beta.threads.create()

    await db_add_thread_user_resource(db, current_user.get("username"), new_thread.id, SIGN_CONVERSATION)

    return CreateThreadResourceResp(id=new_thread.id)


@router.get('/sign', status_code=status.HTTP_200_OK)
async def sign_thread_list(
        current_user: Annotated[str, Depends(get_current_user)],
        db: AsyncIOMotorClient = Depends(get_db),  # type: ignore
):
    logging.info(f'Listing threads for: %s', Config.app_settings.get('openai_assistant'))

    user_resource = await db_get_user_resource(db, current_user.get("username"))

    return ListThreadResourceResp(threads=user_resource.get("sign_threads"))

@router.post('/sign/sign_writing', status_code=status.HTTP_200_OK)
async def sign_thread_writing(
        videoFile: UploadFile = File(...),
        username: str = Form(...),
        answer: str = Form(...),
):
    logging.info(f'Signing writing for: %s', Config.app_settings.get('openai_assistant'))
    logging.info(f'Video file: %s', videoFile.filename)
    logging.info(f'Username: %s', username)

    import uuid
    import os
    from datetime import datetime
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = str(uuid.uuid4())[:8]
    webm_filename = f"sign_video_{timestamp}_{random_str}.webm"
    mp4_filename = f"sign_video_{timestamp}_{random_str}.mp4"

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"current file path: {current_dir}")
    project_root = Path("/server")
    logging.info(f"root path: {project_root}")
    user_upload_dir = project_root / "Video" / "user" / username
    os.makedirs(user_upload_dir, exist_ok=True)

    # save original video
    webm_path = str(user_upload_dir / webm_filename)
    with open(webm_path, "wb") as buffer:
        buffer.write(await videoFile.read())

    mp4_path = str(user_upload_dir / mp4_filename)
    logging.info(f"mp4 path: {mp4_path}")
    try:
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", webm_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-v", "verbose",
            mp4_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        logging.info(f"Transforming succeed: {mp4_path}")
    except Exception as e:
        logging.error(f"Transforming failed: {str(e)}")
        # delete webm file
        try:
            os.remove(webm_path)
        except:
            pass

    logging.info(f"is existed: {os.path.exists(mp4_path)}")

    words = sign_language_recognition(mp4_path)

    try:
        # Create the prompt
        prompt = f"""
                I just received the following sequence of words from sign language recognition: {words}

                This is part of a conversation with an AI assistant. The AI assistant's last response was: "{answer}"

                Please reorganize these sign language words into a complete, fluent sentence that makes sense as a natural follow-up to the assistant's response. 
                Add necessary articles, conjunctions, and prepositions while preserving the original meaning of the recognized words.

                Make sure the sentence sounds natural in the context of this conversation.
                Return only the optimized sentence without any additional explanation.
                """

        # Call GPT model
        response = client.chat.completions.create(
            model="gpt-4",  # Or use another suitable model
            messages=[
                {"role": "system",
                 "content": "You are a language optimization assistant that specializes in combining fragmented words into fluent sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more deterministic answers
            max_tokens=100
        )

        # Get the optimized sentence
        optimized_sentence = response.choices[0].message.content.strip()

        # Log original recognition and optimized result
        logging.info(f"Original recognized words: {words}")
        logging.info(f"Response sentence: {answer}")
        logging.info(f"Optimized sentence: {optimized_sentence}")

        response = optimized_sentence

    except Exception as e:
        logging.error(f"Error optimizing sentence: {str(e)}")
        # If optimization fails, return the original recognition result
        response =  " ".join(words) if isinstance(words, list) else words

    return response

@router.delete('/sign', status_code=status.HTTP_204_NO_CONTENT)
async def sign_thread_delete(
        thread_id: str,
        current_user: Annotated[str, Depends(get_current_user)],
        db: AsyncIOMotorClient = Depends(get_db),  # type: ignore
):
    logging.info(f'Removing thread {thread_id}')

    await db_remove_thread_user_resource(db, current_user.get("username"), thread_id, SIGN_CONVERSATION)

    return None


@router.get('/sign/{thread_id}/messages', status_code=status.HTTP_200_OK)
async def sign_messages_list(
        thread_id,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Listing messages for thread {thread_id}')

    thread_messages = client.beta.threads.messages.list(thread_id, limit=50)

    return {
        'messages': thread_messages.data}  # Replaced model with hardcoded response because new API version breaks the model code

    return ListMessageResourceResp(messages=thread_messages.data)


@router.post('/sign/{thread_id}/messages', status_code=status.HTTP_200_OK)
async def sign_messages_send(
        thread_id,
        content_data: SendMessageResourceReq,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Message to thread {thread_id}')

    thread_message = client.beta.threads.messages.create(
        thread_id,
        role="user",
        content=content_data.content,
    )

    return SendMessageResourceResp(id=thread_message.id)


@router.post('/sign/{thread_id}/messages/assistant', status_code=status.HTTP_200_OK)
async def sign_messages_send(
        thread_id,
        content_data: SendMessageResourceReq,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Into message to thread {thread_id}')

    thread_message = client.beta.threads.messages.create(
        thread_id,
        role="assistant",
        content=content_data.content,
    )

    return SendMessageResourceResp(id=thread_message.id)


@router.post('/sign/{thread_id}/runs', status_code=status.HTTP_200_OK)
async def sign_thread_run(
        thread_id,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Run thread {thread_id} for assistant {Config.app_settings.get("openai_assistant")}')

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=Config.app_settings.get("openai_assistant")
    )

    return RunThreadResp(id=run.id)


@router.get('/sign/{thread_id}/runs/{run_id}', status_code=status.HTTP_200_OK)
async def sign_thread_check(
        thread_id,
        run_id,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Run thread {thread_id} for assistant {Config.app_settings.get("openai_assistant")}')

    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )

    return {'status': run.status,
            'action': run.required_action}  # Replaced model with hardcoded response because new API version breaks the model code

    return RunThreadStatusResp(status=run.status, action=run.required_action)


@router.post('/sign/{thread_id}/runs/{run_id}/submit_tool_outputs', status_code=status.HTTP_204_NO_CONTENT)
async def sign_thread_submit_tool(
        thread_id,
        run_id,
        req_data: SubmitToolsReqText,
        current_user: Annotated[str, Depends(get_current_user)],
):
    logging.info(f'Submit tool outputs for call {req_data.tool_call_id}')

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=[{
            "tool_call_id": req_data.tool_call_id,
            "output": fetch_search_results_text(req_data.prompt)
        }]
    )