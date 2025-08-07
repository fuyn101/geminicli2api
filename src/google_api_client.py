"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import json
import logging
import requests
import asyncio
import time
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest

from .utils import get_user_agent, retry_api_call
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts,
    is_pseudo_streaming_model,
    PSEUDO_STREAMING_HEARTBEAT_INTERVAL
)

class GoogleApiClient:
    """
    A singleton client for interacting with the Google Gemini API.
    Handles credential management, user onboarding, and request signing.
    """
    def __init__(self):
        """
        The client is now stateless. Initialization is handled by the dependency injection system.
        """
        pass

    @retry_api_call(retries=3, delay=1)
    def _make_request(self, url, data, headers, stream=False):
        """Makes the actual HTTP request with retry mechanism."""
        return requests.post(url, data=data, headers=headers, stream=stream)

    def send_request(self, payload: dict, creds, project_id, is_streaming: bool = False) -> Response:
        """
        Send a request to Google's Gemini API using the provided credentials.
        
        Args:
            payload: The request payload in Gemini format.
            creds: The OAuth2 credentials for this request.
            project_id: The Google Cloud project ID for this request.
            is_streaming: Whether this is a streaming request.
            
        Returns:
            FastAPI Response object.
        """
        if not creds or not project_id:
            return Response(
                content=json.dumps({
                    "error": {
                        "message": "Invalid session provided to send_request.",
                        "type": "auth_error",
                        "code": 500
                    }
                }),
                status_code=500,
                media_type="application/json"
            )

        # Build the final payload with project info
        final_payload = {
            "model": payload.get("model"),
            "project": project_id,
            "request": payload.get("request", {})
        }

        # Determine the action and URL based on the new logic
        # Use original model name for pseudo-streaming detection, fallback to base model name
        original_model = payload.get("original_model", payload.get("model", ""))
        model_name = payload.get("model", "")
        use_pseudo_streaming = is_streaming and is_pseudo_streaming_model(original_model)
        
        if is_streaming and not use_pseudo_streaming:
            # For streaming requests with non-pseudo models, use true streaming
            action = "streamGenerateContent"
            target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}?alt=sse"
        else:
            # For non-streaming requests or pseudo-streaming, use non-streaming endpoint
            action = "generateContent"
            target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"

        # Build request headers
        request_headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }

        final_post_data = json.dumps(final_payload)

        # Send the request
        try:
            if is_streaming:
                if use_pseudo_streaming:
                    # For pseudo-streaming models, use pseudo-streaming mode
                    return self._handle_pseudo_streaming(target_url, final_post_data, request_headers)
                else:
                    # For normal streaming models, use true streaming
                    resp = self._make_request(target_url, final_post_data, request_headers, stream=True)
                    return self._handle_streaming_response(resp)
            else:
                # For non-streaming requests, check if keepalive is enabled
                from .config import NONSTREAM_KEEPALIVE_ENABLED
                
                if NONSTREAM_KEEPALIVE_ENABLED:
                    return self._handle_nonstream_keepalive(target_url, final_post_data, request_headers)
                else:
                    # Normal non-streaming mode
                    resp = self._make_request(target_url, final_post_data, request_headers)
                    return self._handle_non_streaming_response(resp)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request to Google API failed after retries: {str(e)}")
            return Response(
                content=json.dumps({"error": {"message": f"Request failed after retries: {str(e)}"}}),
                status_code=500,
                media_type="application/json"
            )
        except Exception as e:
            logging.error(f"Unexpected error during Google API request: {str(e)}")
            return Response(
                content=json.dumps({"error": {"message": f"Unexpected error: {str(e)}"}}),
                status_code=500,
                media_type="application/json"
            )

    def _handle_nonstream_keepalive(self, target_url, final_post_data, request_headers) -> StreamingResponse:
        """Handle non-streaming request with keepalive: send newlines while making API request."""
        from .config import NONSTREAM_KEEPALIVE_INTERVAL
        keepalive_interval = NONSTREAM_KEEPALIVE_INTERVAL
        
        async def keepalive_generator():
            try:
                # Start API request concurrently in a thread pool to avoid blocking
                async def make_api_request():
                    """Make the actual API request."""
                    try:
                        # Run the synchronous _make_request in a thread pool
                        loop = asyncio.get_event_loop()
                        resp = await loop.run_in_executor(
                            None, 
                            self._make_request, 
                            target_url, final_post_data, request_headers, False
                        )
                        return resp
                    except Exception as e:
                        logging.error(f"API request failed: {str(e)}")
                        raise
                
                # Start API request
                api_task = asyncio.create_task(make_api_request())
                
                # Send keepalive messages while waiting for API response
                keepalive_counter = 0
                
                while not api_task.done():
                    # Wait for the keepalive interval or task completion
                    done, pending = await asyncio.wait(
                        [api_task],
                        timeout=keepalive_interval,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # If no task completed, send keepalive message
                    if not done:
                        keepalive_counter += 1
                        yield "\n"  # Send newline as keepalive message
                        continue
                    
                    # If task completed, break the loop
                    break
                
                # Wait for API response to complete
                try:
                    resp = await api_task
                except Exception as e:
                    # Handle API error
                    error_response = {
                        "error": {
                            "message": f"API request failed: {str(e)}",
                            "type": "api_error",
                            "code": 500
                        }
                    }
                    yield json.dumps(error_response, ensure_ascii=False)
                    return
                
                # Handle API response
                if resp.status_code != 200:
                    # Handle error response
                    try:
                        error_data = resp.json()
                        if "error" in error_data:
                            error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                            error_response = {
                                "error": {
                                    "message": error_message,
                                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                                    "code": resp.status_code
                                }
                            }
                            yield json.dumps(error_response, ensure_ascii=False)
                            return
                    except (json.JSONDecodeError, KeyError):
                        pass
                    
                    # Fallback error response
                    error_response = {
                        "error": {
                            "message": f"API error: {resp.status_code}",
                            "type": "api_error",
                            "code": resp.status_code
                        }
                    }
                    yield json.dumps(error_response, ensure_ascii=False)
                    return
                
                # Process successful response
                try:
                    # Force UTF-8 encoding for proper text handling
                    if resp.encoding != 'utf-8':
                        raw_response = resp.content.decode('utf-8').strip()
                    else:
                        raw_response = resp.text.strip()
                    
                    # Parse the complete response
                    parsed_response = json.loads(raw_response)
                    
                    # Check if response has the expected structure
                    if "response" in parsed_response:
                        actual_response = parsed_response["response"]
                    else:
                        actual_response = parsed_response
                    
                    response_json = json.dumps(actual_response, ensure_ascii=False)
                    yield response_json
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "error": {
                            "message": f"Failed to parse response: {str(e)}",
                            "type": "parse_error",
                            "code": 500
                        }
                    }
                    yield json.dumps(error_response, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"Unexpected error during non-streaming keepalive: {str(e)}")
                error_response = {
                    "error": {
                        "message": f"Unexpected error: {str(e)}",
                        "type": "api_error",
                        "code": 500
                    }
                }
                yield json.dumps(error_response, ensure_ascii=False)
        
        return StreamingResponse(
            content=keepalive_generator(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    def _handle_pseudo_streaming(self, target_url, final_post_data, request_headers) -> StreamingResponse:
        """Handle pseudo-streaming: send heartbeats while making API request, then format response as SSE chunks."""
        async def pseudo_stream_generator():
            try:
                # Start API request concurrently in a thread pool to avoid blocking
                async def make_api_request():
                    """Make the actual API request."""
                    try:
                        # Run the synchronous _make_request in a thread pool
                        loop = asyncio.get_event_loop()
                        resp = await loop.run_in_executor(
                            None, 
                            self._make_request, 
                            target_url, final_post_data, request_headers, False
                        )
                        return resp
                    except Exception as e:
                        logging.error(f"API request failed: {str(e)}")
                        raise
                
                # Start API request
                api_task = asyncio.create_task(make_api_request())
                
                # Send heartbeats while waiting for API response
                last_heartbeat_time = time.time()
                
                while not api_task.done():
                    current_time = time.time()
                    
                    # Send heartbeat at configured intervals
                    if current_time - last_heartbeat_time >= PSEUDO_STREAMING_HEARTBEAT_INTERVAL:
                        yield "data: {}\n\n".encode('utf-8')
                        last_heartbeat_time = current_time
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.1)
                
                # Wait for API response to complete
                try:
                    resp = await api_task
                except Exception as e:
                    # Handle API error
                    error_response = {
                        "error": {
                            "message": f"API request failed: {str(e)}",
                            "type": "api_error",
                            "code": 500
                        }
                    }
                    yield f'data: {json.dumps(error_response, ensure_ascii=False)}\n\n'.encode('utf-8')
                    return
                
                # Handle API response
                if resp.status_code != 200:
                    # Handle error response
                    try:
                        error_data = resp.json()
                        if "error" in error_data:
                            error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                            error_response = {
                                "error": {
                                    "message": error_message,
                                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                                    "code": resp.status_code
                                }
                            }
                            yield f'data: {json.dumps(error_response, ensure_ascii=False)}\n\n'.encode('utf-8')
                            return
                    except (json.JSONDecodeError, KeyError):
                        pass
                    
                    # Fallback error response
                    error_response = {
                        "error": {
                            "message": f"API error: {resp.status_code}",
                            "type": "api_error",
                            "code": resp.status_code
                        }
                    }
                    yield f'data: {json.dumps(error_response, ensure_ascii=False)}\n\n'.encode('utf-8')
                    return
                
                # Process successful response
                # Force UTF-8 encoding for proper text handling
                if resp.encoding != 'utf-8':
                    raw_response = resp.content.decode('utf-8').strip()
                else:
                    raw_response = resp.text.strip()
                
                # Parse the complete response
                try:
                    parsed_response = json.loads(raw_response)
                    
                    # Check if response has the expected structure
                    if "response" in parsed_response:
                        actual_response = parsed_response["response"]
                    else:
                        actual_response = parsed_response
                    
                    # Convert complete response to streaming chunks for compatibility
                    if "candidates" in actual_response:
                        for candidate in actual_response["candidates"]:
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                
                                # Send each part as a separate chunk
                                for i, part in enumerate(parts):
                                    chunk_candidate = {
                                        "content": {
                                            "role": candidate["content"]["role"],
                                            "parts": [part]
                                        }
                                    }
                                    
                                    # Add finish reason only to the last chunk
                                    if i == len(parts) - 1 and "finishReason" in candidate:
                                        chunk_candidate["finishReason"] = candidate["finishReason"]
                                    
                                    # Add safety ratings to the last chunk
                                    if i == len(parts) - 1 and "safetyRatings" in candidate:
                                        chunk_candidate["safetyRatings"] = candidate["safetyRatings"]
                                    
                                    chunk_response = {"candidates": [chunk_candidate]}
                                    
                                    # Add metadata to the last chunk
                                    if i == len(parts) - 1:
                                        for key in ["usageMetadata", "modelVersion", "createTime", "responseId"]:
                                            if key in actual_response:
                                                chunk_response[key] = actual_response[key]
                                    
                                    chunk_json = json.dumps(chunk_response, separators=(',', ':'), ensure_ascii=False)
                                    yield f"data: {chunk_json}\n\n".encode('utf-8')
                                    
                                    # Small delay between chunks
                                    await asyncio.sleep(0.01)
                    else:
                        # Fallback: send as single chunk if no candidates
                        response_json = json.dumps(actual_response, separators=(',', ':'), ensure_ascii=False)
                        yield f"data: {response_json}\n\n".encode('utf-8')
                    
                except json.JSONDecodeError as e:
                    # Fallback: if still receiving streaming format, handle it
                    lines = raw_response.split('\n')
                    combined_text = ""
                    combined_thoughts = ""
                    final_finish_reason = None
                    final_safety_ratings = None
                    final_usage_metadata = None
                    final_model_version = None
                    final_create_time = None
                    final_response_id = None
                    
                    # Parse each data: line and collect all content
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('data: '):
                            try:
                                json_part = line[6:]
                                parsed_chunk = json.loads(json_part)
                                
                                if "response" in parsed_chunk:
                                    chunk_response = parsed_chunk["response"]
                                    
                                    # Extract content from candidates
                                    if "candidates" in chunk_response:
                                        for candidate in chunk_response["candidates"]:
                                            if "content" in candidate and "parts" in candidate["content"]:
                                                for part in candidate["content"]["parts"]:
                                                    if "text" in part:
                                                        text_content = part["text"]
                                                        if part.get("thought", False):
                                                            combined_thoughts += text_content
                                                        else:
                                                            combined_text += text_content
                                            
                                            # Keep final metadata
                                            if "finishReason" in candidate:
                                                final_finish_reason = candidate["finishReason"]
                                            if "safetyRatings" in candidate:
                                                final_safety_ratings = candidate["safetyRatings"]
                                    
                                    # Keep metadata from the final chunk
                                    if "usageMetadata" in chunk_response:
                                        final_usage_metadata = chunk_response["usageMetadata"]
                                    if "modelVersion" in chunk_response:
                                        final_model_version = chunk_response["modelVersion"]
                                    if "createTime" in chunk_response:
                                        final_create_time = chunk_response["createTime"]
                                    if "responseId" in chunk_response:
                                        final_response_id = chunk_response["responseId"]
                                
                            except json.JSONDecodeError:
                                continue
                    
                    # Build the final combined response
                    if combined_text or combined_thoughts:
                        final_candidate = {
                            "content": {
                                "role": "model",
                                "parts": []
                            }
                        }
                        
                        if combined_text:
                            final_candidate["content"]["parts"].append({"text": combined_text})
                        if combined_thoughts:
                            final_candidate["content"]["parts"].append({"text": combined_thoughts, "thought": True})
                        
                        if final_finish_reason:
                            final_candidate["finishReason"] = final_finish_reason
                        if final_safety_ratings:
                            final_candidate["safetyRatings"] = final_safety_ratings
                        
                        final_response = {"candidates": [final_candidate]}
                        
                        if final_usage_metadata:
                            final_response["usageMetadata"] = final_usage_metadata
                        if final_model_version:
                            final_response["modelVersion"] = final_model_version
                        if final_create_time:
                            final_response["createTime"] = final_create_time
                        if final_response_id:
                            final_response["responseId"] = final_response_id
                        
                        response_json = json.dumps(final_response, separators=(',', ':'), ensure_ascii=False)
                        yield f"data: {response_json}\n\n".encode('utf-8')
                
            except Exception as e:
                logging.error(f"Unexpected error during pseudo streaming: {str(e)}")
                error_response = {
                    "error": {
                        "message": f"Unexpected error: {str(e)}",
                        "type": "api_error",
                        "code": 500
                    }
                }
                yield f'data: {json.dumps(error_response, ensure_ascii=False)}\n\n'.encode('utf-8')
        
        return StreamingResponse(
            content=pseudo_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )


    def _handle_streaming_response(self, resp) -> StreamingResponse:
        """Handle streaming response from Google API."""
        
        if resp.status_code != 200:
            logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
            error_message = f"Google API error: {resp.status_code}"
            try:
                error_data = resp.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass
            
            async def error_generator():
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                        "code": resp.status_code
                    }
                }
                yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8')
            
            response_headers = {
                "Content-Type": "text/event-stream",
                "Content-Disposition": "attachment",
                "Vary": "Origin, X-Origin, Referer",
                "X-XSS-Protection": "0",
                "X-Frame-Options": "SAMEORIGIN",
                "X-Content-Type-Options": "nosniff",
                "Server": "ESF"
            }
            
            return StreamingResponse(
                error_generator(),
                media_type="text/event-stream",
                headers=response_headers,
                status_code=resp.status_code
            )
        
        async def stream_generator():
            try:
                with resp:
                    # Standard streaming behavior
                    for chunk in resp.iter_lines():
                        if chunk:
                            if not isinstance(chunk, str):
                                chunk = chunk.decode('utf-8', "ignore")
                                
                            if chunk.startswith('data: '):
                                chunk = chunk[len('data: '):]
                                
                                try:
                                    obj = json.loads(chunk)
                                    
                                    if "response" in obj:
                                        response_chunk = obj["response"]
                                        response_json = json.dumps(response_chunk, separators=(',', ':'))
                                        response_line = f"data: {response_json}\n\n"
                                        yield response_line.encode('utf-8', "ignore")
                                        await asyncio.sleep(0)
                                    else:
                                        obj_json = json.dumps(obj, separators=(',', ':'))
                                        yield f"data: {obj_json}\n\n".encode('utf-8', "ignore")
                                except json.JSONDecodeError:
                                    continue
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Streaming request failed: {str(e)}")
                error_response = {
                    "error": {
                        "message": f"Upstream request failed: {str(e)}",
                        "type": "api_error",
                        "code": 502
                    }
                }
                yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")
            except Exception as e:
                logging.error(f"Unexpected error during streaming: {str(e)}")
                error_response = {
                    "error": {
                        "message": f"An unexpected error occurred: {str(e)}",
                        "type": "api_error",
                        "code": 500
                    }
                }
                yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")

        response_headers = {
            "Content-Type": "text/event-stream",
            "Content-Disposition": "attachment",
            "Vary": "Origin, X-Origin, Referer",
            "X-XSS-Protection": "0",
            "X-Frame-Options": "SAMEORIGIN",
            "X-Content-Type-Options": "nosniff",
            "Server": "ESF"
        }
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=response_headers
        )

    def _handle_non_streaming_response(self, resp) -> Response:
        """Handle non-streaming response from Google API."""
        if resp.status_code == 200:
            try:
                google_api_response = resp.text
                if google_api_response.startswith('data: '):
                    google_api_response = google_api_response[len('data: '):]
                google_api_response = json.loads(google_api_response)
                standard_gemini_response = google_api_response.get("response")
                return Response(
                    content=json.dumps(standard_gemini_response),
                    status_code=200,
                    media_type="application/json; charset=utf-8"
                )
            except (json.JSONDecodeError, AttributeError) as e:
                logging.error(f"Failed to parse Google API response: {str(e)}")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("Content-Type")
                )
        else:
            logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
            
            try:
                error_data = resp.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                    error_response = {
                        "error": {
                            "message": error_message,
                            "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                            "code": resp.status_code
                        }
                    }
                    return Response(
                        content=json.dumps(error_response),
                        status_code=resp.status_code,
                        media_type="application/json"
                    )
            except (json.JSONDecodeError, KeyError):
                pass
            
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type")
            )

# Singleton instance
google_api_client = None

def get_google_api_client():
    """
    Lazily initializes and returns the singleton GoogleApiClient instance.
    """
    global google_api_client
    if google_api_client is None:
        # First request: Initializing Google API client...
        google_api_client = GoogleApiClient()
    return google_api_client

def build_gemini_payload_from_openai(openai_payload: dict) -> dict:
    """
    Build a Gemini API payload from an OpenAI-transformed request.
    This is used when OpenAI requests are converted to Gemini format.
    """
    model = openai_payload.get("model")
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)
    
    request_data = {
        "contents": openai_payload.get("contents"),
        "systemInstruction": openai_payload.get("systemInstruction"),
        "cachedContent": openai_payload.get("cachedContent"),
        "tools": openai_payload.get("tools"),
        "toolConfig": openai_payload.get("toolConfig"),
        "safetySettings": safety_settings,
        "generationConfig": openai_payload.get("generationConfig", {}),
    }
    
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    return {
        "model": model,
        "original_model": openai_payload.get("original_model"),  # Preserve original model for pseudo-streaming detection
        "request": request_data
    }


def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request.
    This is used for direct Gemini API calls.
    """
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    
    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}
        
    if "thinkingConfig" not in native_request["generationConfig"]:
        native_request["generationConfig"]["thinkingConfig"] = {}
    
    thinking_budget = get_thinking_budget(model_from_path)
    include_thoughts = should_include_thoughts(model_from_path)
    
    native_request["generationConfig"]["thinkingConfig"]["includeThoughts"] = include_thoughts
    native_request["generationConfig"]["thinkingConfig"]["thinkingBudget"] = thinking_budget
    
    if is_search_model(model_from_path):
        if "tools" not in native_request:
            native_request["tools"] = []
        if not any(tool.get("googleSearch") for tool in native_request["tools"]):
            native_request["tools"].append({"googleSearch": {}})
    
    return {
        "model": get_base_model_name(model_from_path),
        "original_model": model_from_path,  # Preserve original model for pseudo-streaming detection
        "request": native_request
    }