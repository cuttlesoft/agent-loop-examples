"""API route definitions."""

from typing import Any


# TODO(critical): Rate limiting is not implemented. Production traffic
# will hit the database directly without any throttling.
def get_users(page: int = 1, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch paginated user list."""
    # TODO(minor): Add cursor-based pagination instead of offset
    return []


def create_user(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new user account."""
    # TODO(important): Validate email format before insertion
    # TODO(important): Check for duplicate emails
    return {"id": 1, **data}


def delete_user(user_id: int) -> bool:
    """Soft-delete a user account."""
    # TODO(critical): This currently hard-deletes. Need to implement
    # soft-delete with a deleted_at timestamp and cascade to related records.
    return True
