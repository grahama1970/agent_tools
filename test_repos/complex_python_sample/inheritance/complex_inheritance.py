#!/usr/bin/env python3
"""
Example of complex inheritance hierarchies and mixin patterns.
This tests the AST extractor's ability to track inheritance relationships.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypeVar, Generic, Any


# Type variables for generic classes
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class BaseMixin:
    """A mixin that provides common utility methods."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with optional super() call."""
        super().__init__(*args, **kwargs)
        self._mixin_initialized = True
    
    def mixin_method(self) -> str:
        """A method provided by the mixin."""
        return "BaseMixin functionality"


class LoggingMixin:
    """A mixin that provides logging capabilities."""
    
    def log(self, message: str) -> None:
        """Log a message."""
        print(f"LOG: {message}")


class SerializableMixin:
    """A mixin that provides serialization capabilities."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load object from dictionary."""
        for key, value in data.items():
            setattr(self, key, value)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, id: str):
        """Initialize with an ID."""
        self.id = id
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the model."""
        pass


class Repository(Generic[T]):
    """Generic repository for managing collections of items."""
    
    def __init__(self):
        """Initialize an empty repository."""
        self.items: Dict[str, T] = {}
    
    def add(self, item_id: str, item: T) -> None:
        """Add an item to the repository."""
        self.items[item_id] = item
    
    def get(self, item_id: str) -> Optional[T]:
        """Get an item from the repository."""
        return self.items.get(item_id)
    
    def list_all(self) -> List[T]:
        """List all items in the repository."""
        return list(self.items.values())


class UserModel(BaseModel, SerializableMixin, LoggingMixin):
    """
    User model that inherits from multiple classes.
    
    This demonstrates multiple inheritance and mixin usage.
    """
    
    def __init__(self, id: str, username: str, email: str):
        """Initialize user with basic information."""
        BaseModel.__init__(self, id)
        self.username = username
        self.email = email
        self.active = True
        self.log(f"User {username} created")
    
    def validate(self) -> bool:
        """Validate user data."""
        return bool(self.username and self.email and '@' in self.email)
    
    def deactivate(self) -> None:
        """Deactivate the user."""
        self.active = False
        self.log(f"User {self.username} deactivated")


class AdminUser(UserModel):
    """
    Admin user that inherits from UserModel.
    
    This demonstrates multi-level inheritance.
    """
    
    def __init__(self, id: str, username: str, email: str, permissions: List[str]):
        """Initialize admin with permissions."""
        super().__init__(id, username, email)
        self.permissions = permissions
        self.log(f"Admin user {username} created with permissions: {permissions}")
    
    def grant_permission(self, permission: str) -> None:
        """Grant a new permission to the admin."""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.log(f"Permission {permission} granted to {self.username}")


class UserRepository(Repository[UserModel], BaseMixin):
    """
    Repository for managing users with additional mixin functionality.
    
    This demonstrates inheritance from a generic class plus a mixin.
    """
    
    def __init__(self):
        """Initialize repository with mixin."""
        Repository.__init__(self)
        BaseMixin.__init__(self)
    
    def find_by_username(self, username: str) -> Optional[UserModel]:
        """Find a user by username."""
        for user in self.items.values():
            if user.username == username:
                return user
        return None
    
    def find_active_users(self) -> List[UserModel]:
        """Find all active users."""
        return [user for user in self.items.values() if user.active]


# Usage example
if __name__ == "__main__":
    repo = UserRepository()
    
    # Create and add users
    user1 = UserModel("user1", "john", "john@example.com")
    admin1 = AdminUser("admin1", "admin", "admin@example.com", ["users", "content"])
    
    repo.add(user1.id, user1)
    repo.add(admin1.id, admin1)
    
    # Use mixin functionality
    print(repo.mixin_method())
    
    # Use inherited functionality
    active_users = repo.find_active_users()
    print(f"Active users: {len(active_users)}")
    
    # Test serialization
    user_dict = user1.to_dict()
    print(f"User serialized: {user_dict}")