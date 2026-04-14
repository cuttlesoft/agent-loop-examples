"""User data model."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class User:
    """Represents a user account."""

    id: int
    email: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    # TODO(critical): Password is stored as plain text. Must hash with
    # bcrypt or argon2 before this goes anywhere near production.
    password: str = ""
    is_active: bool = True

    def deactivate(self) -> None:
        """Mark user as inactive."""
        self.is_active = False
        # TODO(important): Trigger notification to user's email
        # TODO(important): Revoke active API tokens on deactivation
        # TODO(minor): Update audit log with deactivation reason

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        # TODO(minor): Use a proper serialization library
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "is_active": self.is_active,
        }
