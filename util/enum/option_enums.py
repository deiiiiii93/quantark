"""
Enumeration types for option specifications.
"""
from enum import Enum, auto


class OptionType(Enum):
    """Type of option: Call or Put."""
    CALL = auto()
    PUT = auto()
    
    def __str__(self):
        return self.name.capitalize()


class ExerciseType(Enum):
    """Exercise style of the option."""
    EUROPEAN = auto()
    AMERICAN = auto()
    BERMUDAN = auto()
    
    def __str__(self):
        return self.name.capitalize()

