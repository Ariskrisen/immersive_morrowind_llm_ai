from abc import ABC, abstractmethod

from typing import Optional

from pydantic import BaseModel

from tts.voice import Voice


class TtsBackendRequest(BaseModel):
    text: str
    voice: Voice
    file_path: str
    voice_identifier: Optional[str] = None


class TtsBackendResponse(BaseModel):
    file_path: str

class AbstractTtsBackend(ABC):
    @abstractmethod
    async def convert(self, request: TtsBackendRequest) -> TtsBackendResponse | None:
        pass