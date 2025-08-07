import os
import json
import base64
import time
import logging
import threading
import glob
import requests
from multiprocessing import Value, Lock, Manager
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBasic

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

from .utils import get_user_agent, get_client_metadata
from .config import (
    CLIENT_ID, CLIENT_SECRET, SCOPES, CREDENTIAL_FILE, GEMINI_CREDENTIALS_PATH,
    CODE_ASSIST_ENDPOINT, GEMINI_AUTH_PASSWORD
)

class CredentialManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        with self._lock:
            if hasattr(self, '_initialized'):
                return
            
            self.credentials_pool = []
            # Use multiprocessing Value for cross-process index sharing
            self.current_index = Value('i', 0)
            # Use multiprocessing Lock for cross-process synchronization
            self.rotation_lock = Lock()
            self.cool_down_period = timedelta(minutes=5)
            self._load_credentials()
            self._initialized = True

    def _load_credentials(self):
        if not os.path.isdir(GEMINI_CREDENTIALS_PATH):
            return

        json_files = glob.glob(os.path.join(GEMINI_CREDENTIALS_PATH, '*.json'))
        if not json_files:
            return

        for file_path in json_files:
            try:
                # 首先，从 json 文件中读取 project_id
                with open(file_path, 'r') as f:
                    data = json.load(f)
                project_id = data.get("project_id", "unknown") # 获取 project_id，提供一个备用值

                # 然后，像之前一样加载凭证
                creds = Credentials.from_authorized_user_file(file_path, SCOPES)

                # 将包括 project_id 在内的完整信息添加到池中
                self.credentials_pool.append({
                    "credentials": creds,
                    "file_path": file_path,
                    "project_id": project_id, # 添加 project_id
                    "last_failure": None
                })
            except Exception as e:
                pass
        
        if self.credentials_pool:
            pass
        else:
            pass


    def get_next_credential(self):
        if not self.credentials_pool:
            return None, None, None

        with self.rotation_lock:
            start_index = self.current_index.value
            pool_size = len(self.credentials_pool)

            for i in range(pool_size):
                idx = (start_index + i) % pool_size
                cred_info = self.credentials_pool[idx]
                
                if cred_info["last_failure"] and (datetime.now() - cred_info["last_failure"] < self.cool_down_period):
                    continue

                self.current_index.value = (idx + 1) % pool_size
                
                try:
                    file_path = cred_info.get("file_path") # 在 try 块的开始处获取 file_path
                    creds = cred_info["credentials"]
                    project_id = cred_info.get("project_id")
                    if not project_id:
                        continue

                    if creds.expired and creds.refresh_token:
                        # Credential expired, attempting refresh...
                        creds.refresh(GoogleAuthRequest())
                        # Credential refreshed successfully.
                        cred_info["last_failure"] = None
                    
                    # Using credential for project: '{project_id}' (from {file_path})
                    return creds, project_id, file_path
                except Exception as e:
                    # Failed to refresh credential
                    cred_info["last_failure"] = datetime.now()
                    continue
        
        # All credentials in the pool are currently in cool-down.
        return None, None, None

# --- Global State ---
credentials = None
user_project_id = None
# Use a multiprocessing-safe dictionary to track onboarding status across processes.
# The key is the credential file_path, ensuring onboarding is per-credential.
onboarding_status = None

def get_onboarding_status():
    """Lazily initializes and returns the shared onboarding status dictionary."""
    global onboarding_status
    if onboarding_status is None:
        onboarding_status = Manager().dict()
    return onboarding_status
credentials_from_env = False

credential_manager = None

def get_credential_manager():
    """Lazily initializes and returns the CredentialManager singleton."""
    global credential_manager
    if credential_manager is None:
        credential_manager = CredentialManager()
    return credential_manager
security = HTTPBasic()

def authenticate_user(request: Request):
    """Authenticate the user with multiple methods."""
    # Check for API key in query parameters first (for Gemini client compatibility)
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD:
        return "api_key_user"
    
    # Check for API key in x-goog-api-key header (Google SDK format)
    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD:
        return "goog_api_key_user"
    
    # Check for API key in Authorization header (Bearer token format)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if bearer_token == GEMINI_AUTH_PASSWORD:
            return "bearer_user"
    
    # Check for HTTP Basic Authentication
    if auth_header.startswith("Basic "):
        try:
            encoded_credentials = auth_header[6:]
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8', "ignore")
            username, password = decoded_credentials.split(':', 1)
            if password == GEMINI_AUTH_PASSWORD:
                return username
        except Exception:
            pass
    
    # If none of the authentication methods work
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials. Use HTTP Basic Auth, Bearer token, 'key' query parameter, or 'x-goog-api-key' header.",
        headers={"WWW-Authenticate": "Basic"},
    )

def save_credentials(creds, project_id=None):
    global credentials_from_env
    
    # Don't save credentials to file if they came from environment variable,
    # but still save project_id if provided and no file exists or file lacks project_id
    if credentials_from_env:
        if project_id and os.path.exists(CREDENTIAL_FILE):
            try:
                with open(CREDENTIAL_FILE, "r") as f:
                    existing_data = json.load(f)
                # Only update project_id if it's missing from the file
                if "project_id" not in existing_data:
                    existing_data["project_id"] = project_id
                    with open(CREDENTIAL_FILE, "w") as f:
                        json.dump(existing_data, f, indent=2)
                    pass
            except Exception as e:
                pass
        return
    
    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": creds.scopes if creds.scopes else SCOPES,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    
    if creds.expiry:
        if creds.expiry.tzinfo is None:
            from datetime import timezone
            expiry_utc = creds.expiry.replace(tzinfo=timezone.utc)
        else:
            expiry_utc = creds.expiry
        # Keep the existing ISO format for backward compatibility, but ensure it's properly handled during loading
        creds_data["expiry"] = expiry_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    if project_id:
        creds_data["project_id"] = project_id
    elif os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                existing_data = json.load(f)
                if "project_id" in existing_data:
                    creds_data["project_id"] = existing_data["project_id"]
        except Exception:
            pass
    
    
    # NOTE: File writing is disabled. This is a legacy function that interferes
    # with the multi-credential pool mechanism. Credential files are now managed
    # exclusively by the get_oauth_token.py script.
    # with open(CREDENTIAL_FILE, "w") as f:
    #     json.dump(creds_data, f, indent=2)
    

def get_credentials():
    """
    Gets the next available credential. It strictly prioritizes the credential pool
    if the directory exists, falling back to other methods only if it doesn't.
    """
    manager = get_credential_manager()

    # Strict Check: If the credentials directory exists, we MUST use it.
    if os.path.isdir(GEMINI_CREDENTIALS_PATH):
        # Lazily load credentials if the pool is empty
        if not manager.credentials_pool:
            # Credential pool directory exists. Attempting to load credentials...
            manager._load_credentials()

        if manager.credentials_pool:
            next_cred, project_id, file_path = manager.get_next_credential()
            if next_cred:
                return next_cred, project_id, file_path
            else:
                # Pool exists but all are in cooldown, this is a valid final state.
                # All credentials in the pool are currently in cool-down.
                return None, None, None
        else:
            # Directory exists but loading resulted in an empty pool. This is an error.
            # Credential directory exists but contains no valid credentials.
            return None, None, None

    # Fallback 1: Environment Variable (only if credentials directory does NOT exist)
    env_creds_json = os.getenv("GEMINI_CREDENTIALS")
    if env_creds_json:
        try:
            info = json.loads(env_creds_json)
            creds = Credentials.from_authorized_user_info(info, SCOPES)
            project_id = info.get("project_id", "unknown_env")
            if creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
            # Using credential from environment variable
            return creds, project_id, "environment"
        except Exception as e:
            # Failed to load credentials from environment variable
            pass

    # Fallback 2: Single Credential File (only if directory and env var are not used)
    if os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                info = json.load(f)
            creds = Credentials.from_authorized_user_file(CREDENTIAL_FILE, SCOPES)
            project_id = info.get("project_id", "unknown_file")
            if creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
            # Using credential from single file
            return creds, project_id, "single_file"
        except Exception as e:
            # Failed to load credentials from single file
            pass

    # No valid credentials found.
    return None, None, None

def onboard_user(creds, project_id, file_path):
    """
    Ensures the user is onboarded for a specific credential, matching gemini-cli setupUser behavior.
    This check is now process-safe and specific to each credential file.
    """
    status_dict = get_onboarding_status()
    # If this specific credential (identified by file_path) has already been onboarded, skip.
    if status_dict.get(file_path):
        return

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            raise Exception(f"Failed to refresh credentials during onboarding: {str(e)}")
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    
    load_assist_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(creds, project_id),
    }
    
    try:
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(load_assist_payload),
            headers=headers,
        )
        resp.raise_for_status()
        load_data = resp.json()
        
        tier = None
        if load_data.get("currentTier"):
            tier = load_data["currentTier"]
        else:
            for allowed_tier in load_data.get("allowedTiers", []):
                if allowed_tier.get("isDefault"):
                    tier = allowed_tier
                    break
            
            if not tier:
                tier = {
                    "name": "",
                    "description": "",
                    "id": "legacy-tier",
                    "userDefinedCloudaicompanionProject": True,
                }

        if tier.get("userDefinedCloudaicompanionProject") and not project_id:
            raise ValueError("This account requires setting the GOOGLE_CLOUD_PROJECT env var.")

        if load_data.get("currentTier"):
            status_dict[file_path] = True
            return

        onboard_req_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(creds, project_id),
        }

        while True:
            onboard_resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                data=json.dumps(onboard_req_payload),
                headers=headers,
            )
            onboard_resp.raise_for_status()
            lro_data = onboard_resp.json()

            if lro_data.get("done"):
                status_dict[file_path] = True # Mark this credential as successfully onboarded
                break
            
            time.sleep(5)

    except requests.exceptions.HTTPError as e:
        raise Exception(f"User onboarding failed. Please check your Google Cloud project permissions and try again. Error: {e.response.text if hasattr(e, 'response') else str(e)}")
    except Exception as e:
        raise Exception(f"User onboarding failed due to an unexpected error: {str(e)}")


def get_current_session():
    """
    A FastAPI dependency that provides a fresh, rotated credential session for each request.
    It will attempt to get a working credential up to 3 times before failing.
    """
    last_exception = None
    for attempt in range(3):
        try:
            creds, project_id, file_path = get_credentials()
            if not creds:
                # This counts as a failed attempt if no credential could be retrieved
                last_exception = HTTPException(status_code=503, detail=f"Attempt {attempt + 1}/3: No available credentials.")
                continue

            # Ensure user is onboarded for this specific credential.
            # This might raise an exception, which is a failed attempt.
            onboard_user(creds, project_id, file_path)
            
            # If onboard_user succeeds, we have a working credential.
            return creds, project_id

        except Exception as e:
            # Capture the exception and try the next credential.
            last_exception = e
            continue # Move to the next attempt

    # If all attempts fail, raise the last known exception.
    if isinstance(last_exception, HTTPException):
        # Re-raise the specific HTTPException from get_credentials
        raise last_exception
    elif last_exception:
        # Raise a generic 503 error with the details of the last onboarding failure.
        raise HTTPException(status_code=503, detail=f"All credential attempts failed. Last error: {str(last_exception)}")
    else:
        # This case should ideally not be reached, but as a fallback.
        raise HTTPException(status_code=503, detail="Failed to obtain a valid credential after 3 attempts.")
