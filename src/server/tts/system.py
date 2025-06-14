import os
import subprocess
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field
from util.logger import Logger

from tts.backend.abstract import AbstractTtsBackend, TtsBackendRequest
from tts.backend.elevenlabs import ElevenlabsTtsBackend
from tts.backend.openai import OpenaiTtsBackend
from tts.backend.dummy import DummyTtsBackend
from tts.file_list_rotation import FileListRotation
from tts.request import TtsRequest
from tts.response import TtsResponse
from util.colored_lines import green

logger = Logger(__name__)

class TtsSystem:
    class Config(BaseModel):
        class Ffmpeg(BaseModel):
            path_to_ffmpeg_exe: str
            speed: float = 1.0
            pitch: float = 1.0
            target_char_per_sec: int = 12 

        class Dummy(BaseModel):
            type: Literal['dummy']

        class Elevenlabs(BaseModel):
            type: Literal['elevenlabs']
            elevenlabs: ElevenlabsTtsBackend.Config

        class Openai(BaseModel):
            type: Literal['openai']
            openai: OpenaiTtsBackend.Config

        system: Union[Dummy, Elevenlabs, Openai] = Field(discriminator='type')
        
        output: FileListRotation.Config
        ffmpeg: Optional[Ffmpeg] = Field(default=None)
        sync_print_and_speak: bool = Field(default=False)

    def __init__(self, morrowind_data_files_dir: str, config: Config):
        self._config = config
        sound_output_dir = os.path.join(morrowind_data_files_dir, "Sound", "Vo")
        self._fsrotate = FileListRotation(config.output, sound_output_dir)
        self._backend = self._create_backend()

    async def convert(self, request: TtsRequest) -> TtsResponse | None:
        backend_response = await self._backend.convert(request=TtsBackendRequest(
            text=request.text,
            voice=request.voice,
            file_path=self._fsrotate.get_next_filepath()
        ))
        if backend_response is None:
            return None

        is_pitch_already_applied = False
        if self._config.ffmpeg:
            (path_before_ext, ext) = os.path.splitext(backend_response.file_path)
            file_path_tmp = f"{path_before_ext}_tmp{ext}"

            speed = self._config.ffmpeg.speed
            pitch = self._config.ffmpeg.pitch
            
            tempo = speed / pitch
            
            logger.debug(f"Applying ffmpeg with Speed: {speed}, Pitch: {pitch}")
            
            filter_command = f"asetrate=44100*{pitch},atempo={tempo},aformat=s16,pan=mono|c0=c0+c1"
            
            args = [
                self._config.ffmpeg.path_to_ffmpeg_exe,
                "-i", backend_response.file_path,
                "-filter:a", filter_command,
                "-ar", "44100",  
                "-ac", "1",      
                "-b:a", "64k",   
                file_path_tmp,
                "-y"
            ]
            logger.debug(f"Final FFmpeg command: {args}")

            ffmpeg_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if ffmpeg_process.stdout:
                with ffmpeg_process.stdout:
                    for line in iter(ffmpeg_process.stdout.readline, b''):
                        logger.debug(line.strip())
            exitcode = ffmpeg_process.wait()
            
            if exitcode == 0:
                os.unlink(backend_response.file_path)
                os.rename(file_path_tmp, backend_response.file_path)
                logger.debug(f"FFmpeg re-encoded audio successfully with code {exitcode}")
            else:
                logger.error(f"FFmpeg failed with exit code {exitcode}. Original audio will be used.")

            is_pitch_already_applied = True
        
        return TtsResponse(
            file_path=backend_response.file_path,
            is_pitch_already_applied=is_pitch_already_applied
        )

    def _create_backend(self) -> AbstractTtsBackend:
        system = self._config.system
        backend: AbstractTtsBackend
        if system.type == 'dummy':
            logger.info(f"Text-to-speech system is set to {green('dummy')}")
            backend = DummyTtsBackend()
        elif system.type == 'elevenlabs':
            logger.info(f"Text-to-speech system is set to {green('ElevenLabs')}")
            backend = ElevenlabsTtsBackend(system.elevenlabs)
        elif system.type == 'openai':
            logger.info(f"Text-to-speech system is set to {green('OpenAI')}")
            backend = OpenaiTtsBackend(system.openai)
        else:
            raise Exception(f"Unknown text-to-speech system '{system.type}'")
        return backend