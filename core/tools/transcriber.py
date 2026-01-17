import os
import json
import time
import torch
import logging
import yt_dlp
import warnings
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Any, Optional
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from config import ProjectConfig, logger
from utils.memory import release_vram
from utils.gpu_lock import GPULock

# Silence annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

class VideoTranscriber:
    """
    Advanced Media Transcription & Diarization Pipeline (ETL Optimized).
    
    Refactored for RTX 3060 (12GB VRAM):
    - Models are loaded ON-DEMAND only.
    - Aggressive VRAM cleanup after transcription.
    - Outputs raw JSON to INBOX_DIR for asynchronous processing.
    """

    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.logger = logging.getLogger("VideoTranscriber")
        
        self.logger.info(
            f"Initialized VideoTranscriber (Stateless Mode). Device: {self.device}", 
            extra={"tags": "MEDIA-INIT"}
        )

    def download_video(self, url: str, progress_callback=None) -> Dict[str, Any]:
        """Downloads audio and returns metadata dict."""
        output_dir = str(ProjectConfig.TEMP_DIR)
        out_template = os.path.join(output_dir, '%(id)s.%(ext)s')

        def progress_hook(d):
            if d['status'] == 'downloading':
                if progress_callback:
                    percent = d.get('_percent_str', 'N/A')
                    eta = d.get('_eta_str', 'N/A')
                    progress_callback(f"Pobieranie: {percent} | ETA: {eta}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'restrictfilenames': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'progress_hooks': [progress_hook],
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                final_path = os.path.join(output_dir, f"{info['id']}.mp3")
                
                metadata = {
                    "id": info.get('id'),
                    "title": info.get('title', 'Unknown Title'),
                    "uploader": info.get('uploader', 'Unknown'),
                    "duration": info.get('duration', 0),
                    "local_path": final_path,
                    "url": url
                }
                
                self.logger.info(f"Downloaded media: {final_path}", extra={"tags": "MEDIA-DOWNLOAD"})
                return metadata
        except Exception as e:
            self.logger.error(f"Download failed: {e}", extra={"tags": "MEDIA-ERROR"})
            raise

    def _try_fetch_transcript(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to fetch transcript directly from YouTube using unofficial API.
        Returns formatted payload or None if not available/failed.
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pl', 'en'])
            
            # Format to same structure as Whisper output
            full_text_parts = []
            segments_list = []
            
            for item in transcript:
                text = item['text'].strip()
                full_text_parts.append(text)
                segments_list.append({
                    "start": item['start'],
                    "end": item['start'] + item['duration'],
                    "text": text
                })
                
            self.logger.info(f"Fetched transcript for {video_id} via API", extra={"tags": "TRANSCRIPT-API"})
            return {
                "text": " ".join(full_text_parts),
                "segments": segments_list
            }
            
        except Exception as e:
            self.logger.warning(f"Could not fetch transcript via API for {video_id}: {e}", extra={"tags": "TRANSCRIPT-API-MISS"})
            return None

    def process_to_inbox(self, url: str, progress_callback=None) -> str:
        """
        Main Pipeline: Download -> Transcribe -> Save to Inbox -> Release VRAM.
        Returns the path to the saved JSON file.
        """
        try:
            # 1. Download
            meta = self.download_video(url, progress_callback)
            
            # OPTIMIZATION: Try to fetch transcript via API first
            transcript_data = self._try_fetch_transcript(meta['id'])
            
            if not transcript_data:
                self.logger.info("Falling back to Whisper...", extra={"tags": "FALLBACK"})
                audio_path = meta['local_path']
                # 2. Transcribe (Load -> Run -> Unload)
                transcript_data = self._run_transcription_isolated(audio_path, progress_callback)
            else:
                if progress_callback: progress_callback("Pobrano transkrypt z YouTube API (pominięto Whisper)") 
            
            # 3. Construct Payload
            payload = {
                "meta": meta,
                "content": transcript_data['text'],
                "segments": transcript_data['segments'],
                "processed_at": time.time(),
                "status": "ready_for_refinery"
            }

            # 4. Save to INBOX
            safe_title = "".join([c for c in meta['id'] if c.isalnum() or c in ('-','_')])
            output_filename = f"{safe_title}.json"
            output_path = ProjectConfig.INBOX_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved payload to Inbox: {output_path}", extra={"tags": "ETL-LOAD"})
            return str(output_path)

        except Exception as e:
            self.logger.error(f"ETL Process Failed: {e}", extra={"tags": "FATAL"})
            raise

    def process_local_file(self, file_path: str, progress_callback=None) -> str:
        """
        Process a local audio file directly (Watchdog mode).
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # 1. Metadata
            meta = {
                "id": path.stem,
                "title": path.stem,
                "uploader": "Local User",
                "duration": 0, # Could be extracted with ffmpeg/pydub if needed
                "local_path": str(path),
                "url": "local"
            }

            # 2. Transcribe
            transcript_data = self._run_transcription_isolated(str(path), progress_callback)

            # 3. Construct Payload
            payload = {
                "meta": meta,
                "content": transcript_data['text'],
                "segments": transcript_data['segments'],
                "processed_at": time.time(),
                "status": "ready_for_refinery"
            }

            # 4. Save to INBOX (System Inbox for processing)
            safe_title = "".join([c for c in meta['id'] if c.isalnum() or c in ('-','_')])
            output_filename = f"{safe_title}.json"
            output_path = ProjectConfig.INBOX_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved local payload to Inbox: {output_path}", extra={"tags": "ETL-LOAD-LOCAL"})
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Local Process Failed: {e}", extra={"tags": "FATAL"})
            raise

    def _run_transcription_isolated(self, audio_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Runs Whisper in an isolated manner. Loads model, processes, then forces unload.
        """
        model = None
        lock = GPULock()
        try:
            with lock:
                # Ensure VRAM is clean before starting
                release_vram()
                
                self.logger.info(f"Loading Whisper ({self.model_size})...", extra={"tags": "MODEL-LOAD"})
                if progress_callback: progress_callback("Ładowanie modelu Whisper...")
                
                model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                
                self.logger.info("Transcribing...", extra={"tags": "WHISPER"})
                if progress_callback: progress_callback("Transkrypcja w toku...")
                
                segments_gen, info = model.transcribe(audio_path, vad_filter=True)
                
                segments_list = []
                full_text_parts = []
                
                total_duration = info.duration
                for segment in segments_gen:
                    text = segment.text.strip()
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": text
                    })
                    full_text_parts.append(text)
                    
                    # Optional visual feedback
                    if progress_callback and total_duration > 0:
                        percent = int((segment.end / total_duration) * 100)
                        progress_callback(f"Transkrypcja: {percent}%")

                return {
                    "text": " ".join(full_text_parts),
                    "segments": segments_list
                }

        except Exception as e:
            raise e
        finally:
            # CRITICAL: Clean up
            if model:
                del model
            self.logger.info("Unloaded Whisper.", extra={"tags": "MODEL-UNLOAD"})
            release_vram()

    # Note: Diarization temporarily removed to focus on Whisper stability in Phase 1. 
    # Can be re-added as a separate isolated step in _run_diarization_isolated if needed.