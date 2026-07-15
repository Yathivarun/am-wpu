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

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
