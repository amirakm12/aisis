"""
Model Download and Update System
Handles downloading, updating, and managing model files with resume capability
"""

import os
import hashlib
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from tqdm.asyncio import tqdm
from loguru import logger
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DownloadProgress:
    """Download progress information"""
    total_size: int
    downloaded: int
    speed: float  # bytes/second
    eta: float  # seconds
    status: str
    timestamp: datetime


class ModelDownloader:
    """Handles model downloading and updating with resume capability"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = storage_dir / ".temp"
        self.temp_dir.mkdir(exist_ok=True)
        self._active_downloads: Dict[str, asyncio.Task] = {}
        self._progress_callbacks: Dict[str, Callable[[DownloadProgress], None]] = {}

    async def download_model(
        self,
        model_id: str,
        version: str,
        url: str,
        expected_hash: str,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> bool:
        """
        Download a model with resume capability and progress tracking
        
        Args:
            model_id: Unique identifier for the model
            version: Version string
            url: Download URL
            expected_hash: Expected SHA-256 hash of the file
            chunk_size: Download chunk size in bytes
            progress_callback: Optional callback for progress updates
        
        Returns:
            bool: True if download successful, False otherwise
        """
        download_id = f"{model_id}-{version}"
        if download_id in self._active_downloads:
            logger.warning(f"Download already in progress for {download_id}")
            return False

        if progress_callback:
            self._progress_callbacks[download_id] = progress_callback

        try:
            # Start download task
            task = asyncio.create_task(
                self._download_with_resume(
                    model_id,
                    version,
                    url,
                    expected_hash,
                    chunk_size
                )
            )
            self._active_downloads[download_id] = task
            return await task

        except Exception as e:
            logger.error(f"Error downloading {download_id}: {e}")
            return False

        finally:
            self._active_downloads.pop(download_id, None)
            self._progress_callbacks.pop(download_id, None)

    async def _download_with_resume(
        self,
        model_id: str,
        version: str,
        url: str,
        expected_hash: str,
        chunk_size: int
    ) -> bool:
        """Internal method to handle download with resume capability"""
        temp_file = self.temp_dir / f"{model_id}-{version}.part"
        final_file = self.storage_dir / f"{model_id}-{version}"
        download_id = f"{model_id}-{version}"

        # Check if final file exists and is valid
        if final_file.exists():
            if await self._verify_hash(final_file, expected_hash):
                logger.info(f"File already exists and hash matches for {download_id}")
                return True
            else:
                logger.warning(f"Existing file hash mismatch for {download_id}, redownloading")
                final_file.unlink()

        # Get file size and resume position
        initial_size = temp_file.stat().st_size if temp_file.exists() else 0
        headers = {'Range': f'bytes={initial_size}-'} if initial_size > 0 else {}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if not response.ok:
                        logger.error(f"Failed to download {download_id}: {response.status}")
                        return False

                    total_size = int(response.headers.get('content-length', 0))
                    if initial_size > 0:
                        total_size += initial_size

                    mode = 'ab' if initial_size > 0 else 'wb'
                    async with aiofiles.open(temp_file, mode) as f:
                        downloaded = initial_size
                        start_time = datetime.now()

                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                            downloaded += len(chunk)

                            # Update progress
                            if download_id in self._progress_callbacks:
                                elapsed = (datetime.now() - start_time).total_seconds()
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                eta = (total_size - downloaded) / speed if speed > 0 else 0
                                
                                progress = DownloadProgress(
                                    total_size=total_size,
                                    downloaded=downloaded,
                                    speed=speed,
                                    eta=eta,
                                    status="downloading",
                                    timestamp=datetime.now()
                                )
                                self._progress_callbacks[download_id](progress)

            # Verify downloaded file
            if await self._verify_hash(temp_file, expected_hash):
                # Move to final location
                temp_file.rename(final_file)
                logger.info(f"Successfully downloaded and verified {download_id}")
                return True
            else:
                logger.error(f"Hash verification failed for {download_id}")
                temp_file.unlink()
                return False

        except Exception as e:
            logger.error(f"Error during download of {download_id}: {e}")
            return False

    async def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file hash"""
        if not file_path.exists():
            return False

        try:
            sha256_hash = hashlib.sha256()
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest() == expected_hash

        except Exception as e:
            logger.error(f"Error verifying hash for {file_path}: {e}")
            return False

    def get_download_progress(self, model_id: str, version: str) -> Optional[DownloadProgress]:
        """Get current download progress"""
        download_id = f"{model_id}-{version}"
        if download_id not in self._active_downloads:
            return None

        task = self._active_downloads[download_id]
        if task.done():
            return None

        temp_file = self.temp_dir / f"{model_id}-{version}.part"
        if not temp_file.exists():
            return None

        downloaded = temp_file.stat().st_size
        return DownloadProgress(
            total_size=-1,  # Unknown until download starts
            downloaded=downloaded,
            speed=0.0,  # Cannot calculate without active download
            eta=0.0,
            status="pending" if not task.cancelled() else "cancelled",
            timestamp=datetime.now()
        )

    async def cancel_download(self, model_id: str, version: str) -> bool:
        """Cancel an active download"""
        download_id = f"{model_id}-{version}"
        if download_id not in self._active_downloads:
            return False

        task = self._active_downloads[download_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up temp file
        temp_file = self.temp_dir / f"{model_id}-{version}.part"
        if temp_file.exists():
            temp_file.unlink()

        self._active_downloads.pop(download_id, None)
        self._progress_callbacks.pop(download_id, None)
        return True

    async def cleanup(self) -> None:
        """Clean up temporary files and cancel active downloads"""
        # Cancel all active downloads
        for download_id, task in self._active_downloads.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clear dictionaries
        self._active_downloads.clear()
        self._progress_callbacks.clear()

        # Remove temp directory and its contents
        if self.temp_dir.exists():
            for file in self.temp_dir.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error removing temp file {file}: {e}")
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                logger.error(f"Error removing temp directory: {e}") 