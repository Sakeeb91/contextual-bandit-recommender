"""Base class for contextual bandit algorithms."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseBandit(ABC):
    """
    Abstract base class for contextual bandit algorithms.

    A contextual bandit learns to select actions (arms) based on
    context information to maximize cumulative reward.

    Attributes:
        n_arms: Number of available actions/arms
        context_dim: Dimension of the context vector
    """

    def __init__(self, n_arms: int, context_dim: int) -> None:
        """
        Initialize the bandit.

        Args:
            n_arms: Number of arms (actions) available
            context_dim: Dimension of context feature vectors
        """
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")

        self.n_arms = n_arms
        self.context_dim = context_dim

    @abstractmethod
    def select_arm(self, context: NDArray[np.float64]) -> int:
        """
        Select an arm given the current context.

        Args:
            context: Feature vector of shape (context_dim,)

        Returns:
            Index of the selected arm (0 to n_arms-1)
        """
        pass

    @abstractmethod
    def update(
        self,
        context: NDArray[np.float64],
        arm: int,
        reward: float
    ) -> None:
        """
        Update the model after observing a reward.

        Args:
            context: Feature vector that was used for selection
            arm: The arm that was selected
            reward: The observed reward (typically 0 to 1)
        """
        pass

    @abstractmethod
    def get_arm_values(
        self,
        context: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Get expected values for all arms given context.

        Args:
            context: Feature vector of shape (context_dim,)

        Returns:
            Array of shape (n_arms,) with expected value for each arm
        """
        pass


if __name__ == "__main__":
    # Quick sanity check - this should fail since BaseBandit is abstract
    try:
        b = BaseBandit(10, 5)
        print("ERROR: Should not be able to instantiate abstract class")
    except TypeError as e:
        print(f"OK: Cannot instantiate abstract class - {e}")
