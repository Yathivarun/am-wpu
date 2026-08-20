"""HTTP client utilities with retry logic."""

import logging

import httpx

logger = logging.getLogger(__name__)


class HTTPClient:
    """HTTP client with retry logic for API requests."""

    def __init__(self, timeout: float = 10.0, max_retries: int = 3):
        """
        Initialize the HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def post(
        self,
        url: str,
        data: dict | None = None,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        """
        Send a POST request with retry logic.

        Args:
            url: The URL to send the request to
            data: Form data to send (application/x-www-form-urlencoded)
            json: JSON payload to send (application/json)
            headers: Optional HTTP headers

        Returns:
            Response JSON as a dictionary

        Raises:
            httpx.HTTPError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.post(url, data=data, json=json, headers=headers)
                response.raise_for_status()

                # Log successful response details
                logger.info(
                    f"POST {url} - Status: {response.status_code} - "
                    f"Headers: {dict(response.headers)}"
                )

                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")

        logger.error(f"All retry attempts failed for {url}")
        raise last_error

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        """
        Send a GET request with retry logic.

        Args:
            url: The URL to send the request to
            params: Query parameters
            headers: Optional HTTP headers

        Returns:
            Response JSON as a dictionary

        Raises:
            httpx.HTTPError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.get(url, params=params, headers=headers)
                response.raise_for_status()

                # Log successful response details
                logger.info(
                    f"GET {url} - Status: {response.status_code} - "
                    f"Headers: {dict(response.headers)}"
                )

                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"GET request failed (attempt {attempt + 1}/{self.max_retries}): {e}")

        logger.error(f"All retry attempts failed for {url}")
        raise last_error

    def get_json_once(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict | None:
        """
        Best-effort single-attempt GET returning the decoded JSON body.

        The asset listing endpoints are queried from the recognition thread,
        and a registration with no SAU media legitimately 404s. Going through
        get() would spend max_retries × timeout (30s by default) re-asking a
        question the server has already answered definitively, stalling
        recognition for every such visitor. One attempt, no retry, never
        raises — a failure just means "no assets this time", and the next
        recognition of the same person tries again anyway.

        Returns:
            The decoded JSON object, or None on any failure (network error,
            non-2xx status, non-JSON body).
        """
        try:
            response = self._client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.info(f"Asset listing unavailable at {url}: {e}")
            return None

    def download_binary(
        self,
        url: str,
        headers: dict | None = None,
    ) -> bytes | None:
        """
        Best-effort single-attempt GET returning raw bytes.

        Unlike post()/get(), this never raises and never retries: callers
        fetch optional assets (cutouts, videos) where a missing or
        unreachable file is a normal outcome to be skipped, not an error to
        be recovered from. Returns None on any failure — network error,
        non-2xx status, or anything else.

        Args:
            url: The URL to download
            headers: Optional HTTP headers (e.g. If-None-Match for
                conditional requests)

        Returns:
            Response body as bytes, or None on any failure.
        """
        try:
            response = self._client.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.warning(f"Binary download failed for {url}: {e}")
            return None

    def head(self, url: str, headers: dict | None = None) -> dict | None:
        """
        Best-effort HEAD request returning the response headers.

        Used to revalidate a cached asset (ETag / Last-Modified) without
        re-downloading its body. Never raises — returns None on any failure,
        which callers treat as "can't revalidate, keep what we have".
        """
        try:
            response = self._client.head(url, headers=headers)
            response.raise_for_status()
            return dict(response.headers)
        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
