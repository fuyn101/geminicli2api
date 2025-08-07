import platform
from .config import CLI_VERSION

def get_user_agent():
    """Generate User-Agent string matching gemini-cli format."""
    version = CLI_VERSION
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{version} ({system}; {arch})"

def get_platform_string():
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()
    
    # Map to gemini-cli platform format
    if system == "DARWIN":
        if arch in ["ARM64", "AARCH64"]:
            return "DARWIN_ARM64"
        else:
            return "DARWIN_AMD64"
    elif system == "LINUX":
        if arch in ["ARM64", "AARCH64"]:
            return "LINUX_ARM64"
        else:
            return "LINUX_AMD64"
    elif system == "WINDOWS":
        return "WINDOWS_AMD64"
    else:
        return "PLATFORM_UNSPECIFIED"

def get_client_metadata(creds, project_id=None):
    """Gets the client metadata for API requests."""
    # creds object is now passed directly to this function.
    token = creds.token if creds else ""
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": get_platform_string(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }

import asyncio
import httpx
from functools import wraps

def retry_api_call(retries=3, delay=1):
    """
    A decorator to retry an async function call if it raises an httpx.RequestError.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except httpx.RequestError as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise
        return wrapper
    return decorator
