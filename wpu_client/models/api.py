"""API models for face recognition requests and responses."""


from pydantic import BaseModel, Field


class IdentifyRequest(BaseModel):
    """Request model for the identify API endpoint."""

    type: str = Field(description="Type of identification (e.g., 'face')")
    n: int = Field(description="Number of results to return")
    model: str = Field(
        default="sface",
        description="Recognition model gallery to query (sface=128D, auraface=512D). "
        "Required to disambiguate which 128D gallery a probe vector belongs to, "
        "since more than one 128D embedding space exists server-side.",
    )
    face_vector: str = Field(description="Comma-separated face vector string (128 floats for SFace)")

    @classmethod
    def from_vector_list(
        cls, type: str, n: int, face_vector: list[float], model: str = "sface"
    ) -> "IdentifyRequest":
        """
        Create request from a list of floats.

        Args:
            type: Identification type
            n: Number of results
            face_vector: List of float values
            model: Recognition model gallery to query (default "sface")

        Returns:
            IdentifyRequest instance with comma-separated string
        """
        vector_str = ",".join(str(v) for v in face_vector)
        return cls(type=type, n=n, model=model, face_vector=vector_str)


class IdentifyResponse(BaseModel):
    """Response model from the identify API endpoint."""

    registration_id: str | None = None
    name: str | None = None
    preferred_name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    distance: float | None = None
    confidence: float | None = None
    match_type: str | None = None
    message: str | None = None

    @property
    def person_name(self) -> str:
        """Get person name (alias for name field)."""
        return self.name or "Unknown"

    @property
    def success(self) -> bool:
        """Check if identification was successful."""
        return self.name is not None


class WPULImagesResponse(BaseModel):
    """Response model for WPU images endpoint."""

    signed_urls: list[str] = Field(
        default_factory=list, description="List of signed image URLs"
    )

    @property
    def has_images(self) -> bool:
        """Check if there are any images."""
        return len(self.signed_urls) > 0
