import asyncio
import time
from typing import Optional
from util.logger import Logger
from threading import Lock, Thread
from pydantic import BaseModel

from openai import OpenAI

from tts.backend.abstract import AbstractTtsBackend, TtsBackendRequest, TtsBackendResponse
from tts.voice import Voice

logger = Logger(__name__)

class _Request(BaseModel):
    request_id: int
    text: str
    file_path: str


class OpenaiTtsBackend(AbstractTtsBackend):

    class Config(BaseModel):
        api_key: str
        api_base: Optional[str] = None
        max_wait_time_sec: float

        tts_model: str = "tts-1"
        tts_voice: str = "alloy"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        

        self._openai = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )


        self._max_wait_time_sec = config.max_wait_time_sec
        self._requests_lock = Lock()
        self._requests: list[_Request] = []
        self._next_request_id = 1
        self._responses_lock = Lock()
        self._responses: dict[int, TtsBackendResponse] = {}
        self._loop = asyncio.get_event_loop()

        thread = Thread(target=self._convert_thread, args=[0])
        thread.daemon = True 
        thread.start()

    async def convert(self, request: TtsBackendRequest) -> TtsBackendResponse | None:

        race = request.voice.race_id.lower().replace(" ", "_")
        gender = "female" if request.voice.female else "male"
        voice_identifier = f"{race}_{gender}"
        logger.info(f"[openai.py] Identified voice type: {voice_identifier}")
        

        text_with_id = f"VOICE_ID:{voice_identifier}|||{request.text}"

        internal_request = _Request(
            request_id=self._next_request_id,
            text=text_with_id, 
            file_path=request.file_path,
        )
        self._next_request_id += 1
        
        with self._requests_lock:
            self._requests.append(internal_request)

        t0 = time.time()
        while True:
            dt = time.time() - t0
            if dt > self._max_wait_time_sec:
                logger.error(f"Timeout waiting for TTS result for text: '{request.text}'")
                break
            with self._responses_lock:
                if internal_request.request_id in self._responses:
                    return self._responses.pop(internal_request.request_id)
            await asyncio.sleep(1.0 / 20.0)
        return None

    def _convert_thread(self, thread_index: int):
        """
        Фоновый поток, который в цикле берет запросы из очереди и обрабатывает их.
        Этот метод мы не меняем.
        """
        Logger.set_ctx(f"openai_thread_{thread_index}")
        logger = Logger(__name__ + f".thread.{thread_index}")
        logger.debug("Thread started")

        while self._loop.is_running():
            request_to_process = None
            with self._requests_lock:
                if self._requests:
                    request_to_process = self._requests.pop(0)

            if request_to_process:
                try:
                    response = self._handle_request_in_thread(request_to_process)
                    with self._responses_lock:
                        self._responses[request_to_process.request_id] = response
                except Exception as error:
                    logger.error(f"TTS conversion in thread failed: {error}")
                    logger.debug(f"Request data: {request_to_process}")

            time.sleep(1.0 / 10.0)

        logger.debug("Thread stopped")

    def _handle_request_in_thread(self, request: _Request) -> TtsBackendResponse:
        """
        Эта функция выполняет реальный вызов к API нашего сервера.
        Она получает уже готовый текст с "зашитым" ID голоса.
        """
        logger.debug(f"Handling request in thread {request.request_id}")
        
        response = self._openai.audio.speech.create(
            model=self._config.tts_model,
            voice=self._config.tts_voice,
            input=request.text, 
            response_format="mp3"
        )
        
        response.stream_to_file(request.file_path)
        
        logger.debug(f"Handling request completed {request.request_id}")
        return TtsBackendResponse(file_path=request.file_path)