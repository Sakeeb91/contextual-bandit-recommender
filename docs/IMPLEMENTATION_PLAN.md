# Implementation Plan: Contextual Bandit Recommender

## Expert Role

**ML Engineer (Recommendation Systems Specialist)**

This role was selected because the project requires:
- Deep understanding of multi-armed bandit algorithms (Thompson Sampling, LinUCB)
- Bayesian inference expertise for uncertainty quantification
- Knowledge of collaborative filtering for cold-start handling
- Offline policy evaluation techniques (inverse propensity scoring)
- Production ML system design patterns

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ /recommend  │  │  /feedback  │  │  /explain   │  │  /evaluate          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      RecommendationEngine                               │ │
│  │   - Coordinates all components                                         │ │
│  │   - Handles request routing                                            │ │
│  │   - Manages cold-start detection                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE COMPONENTS                                    │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │ Feature Engine   │    │  Bandit Policy   │    │  Explanation Gen     │   │
│  │                  │    │                  │    │                      │   │
│  │ - User context   │───▶│ - Thompson       │───▶│ - Feature contrib    │   │
│  │ - Item features  │    │   Sampling       │    │ - Uncertainty viz    │   │
│  │ - CF embeddings  │    │ - LinUCB-style   │    │ - Human-readable     │   │
│  │ - Normalization  │    │ - Arm selection  │    │                      │   │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────────┘   │
│                                   │                                          │
│  ┌──────────────────┐    ┌────────▼─────────┐    ┌──────────────────────┐   │
│  │ Cold-Start       │    │  Reward Model    │    │  Policy Evaluator    │   │
│  │ Handler          │    │                  │    │                      │   │
│  │                  │───▶│ - Bayesian Ridge │    │ - IPS estimator      │   │
│  │ - User warmup    │    │ - Uncertainty    │    │ - Offline replay     │   │
│  │ - Item warmup    │    │ - Online update  │    │ - Regret tracking    │   │
│  │ - CF embeddings  │    │                  │    │                      │   │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────────┘   │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │ Replay Buffer    │    │  Model Storage   │    │  Interaction Log     │   │
│  │                  │    │                  │    │                      │   │
│  │ - FIFO queue     │    │ - Bayesian params│    │ - User actions       │   │
│  │ - Prioritized    │    │ - CF matrices    │    │ - Rewards            │   │
│  │ - Batch sampling │    │ - JSON/Pickle    │    │ - Propensities       │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
│                                                                              │
│                         Storage: SQLite + JSON files                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. REQUEST FLOW:
   User Request → API → Feature Engineering → Bandit Policy → Response
                              ↓
                        Cold-Start Check
                              ↓
                        CF Embeddings (if new)

2. FEEDBACK FLOW:
   Reward Signal → API → Replay Buffer → Batch Update → Reward Model
                              ↓
                        Propensity Logging (for offline eval)

3. EVALUATION FLOW:
   Historical Data → IPS Estimator → Policy Score → Decision Report
```

---

## Technology Selection

| Component | Technology | Rationale | Tradeoffs | Fallback |
|-----------|------------|-----------|-----------|----------|
| **Language** | Python 3.9+ | Junior-friendly, rich ML ecosystem, type hints | Slower than compiled languages | N/A (core requirement) |
| **Linear Algebra** | NumPy 1.24+, SciPy 1.10+ | Industry standard, well-documented, vectorized ops | Memory for large matrices | Pure Python (slower) |
| **Bayesian Ridge** | scikit-learn 1.3+ | Built-in uncertainty via `return_std`, no custom math | Less flexible than PyMC | Manual implementation |
| **CF Embeddings** | scikit-learn TruncatedSVD | Simple matrix factorization, no GPU needed | Limited to linear | Surprise library |
| **Data Handling** | pandas 2.0+ | Beginner-friendly, excellent docs | Memory overhead | Pure Python dicts |
| **API Framework** | FastAPI 0.100+ | Auto-docs, type validation, async support | Learning curve for async | Flask |
| **Testing** | pytest 7.0+ | Simple syntax, fixtures, parametrization | None significant | unittest |
| **Storage** | SQLite + JSON | Zero-cost, serverless, portable | Not for high concurrency | PostgreSQL (if scaling) |
| **Serialization** | pickle + JSON | Native Python, simple | Pickle security risks | joblib |

### Key Design Decisions

1. **Bayesian Ridge over custom Thompson Sampling**: scikit-learn's `BayesianRidge` provides `return_std=True` which gives us the uncertainty we need for exploration without implementing Bayesian inference from scratch.

2. **TruncatedSVD over deep learning embeddings**: For a junior developer with laptop-only compute, matrix factorization is interpretable, fast, and requires no GPU.

3. **SQLite over cloud databases**: Zero cost, no setup, portable. Can migrate to PostgreSQL later if needed.

4. **FastAPI over Flask**: Better for learning modern Python patterns (type hints, async), automatic OpenAPI docs help testing.

---

## Phased Implementation Plan

### Phase 1: Core Bandit Algorithm (Foundation)

**Objective**: Implement a working Thompson Sampling bandit with synthetic data.

**Scope**:
- `src/bandit/__init__.py` - Package init
- `src/bandit/base.py` - Abstract base class for bandits
- `src/bandit/thompson.py` - Thompson Sampling implementation
- `src/models/__init__.py` - Package init
- `src/models/bayesian_ridge.py` - Wrapper around sklearn BayesianRidge
- `tests/test_bandit_basic.py` - Basic functionality tests

**Deliverables**:
1. `ContextualBandit` class that can:
   - Accept context vectors and select arms
   - Update beliefs based on rewards
   - Return uncertainty estimates
2. Synthetic experiment showing regret decreasing over time

**Verification**:
```bash
pytest tests/test_bandit_basic.py -v
python -c "from src.bandit import ContextualBandit; print('Import OK')"
```

**Technical Challenges**:
- Understanding Bayesian Ridge coefficient covariance for sampling
- Numerical stability with matrix inversions (use `scipy.linalg.solve` not `np.linalg.inv`)

**Debugging Scenarios**:
- If regret doesn't decrease: Check reward model update frequency
- If NaN values appear: Add small regularization (alpha) to prevent singular matrices

**Time Estimate**: 3-4 hours (cut: skip visualization, focus on core math)

**Definition of Done**:
- [ ] `ContextualBandit.select_arm(context)` returns valid arm index
- [ ] `ContextualBandit.update(context, arm, reward)` updates model
- [ ] Cumulative regret decreases on synthetic linear problem
- [ ] All tests pass

**Code Skeleton**:

```python
# src/bandit/base.py
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class BaseBandit(ABC):
    """Abstract base class for contextual bandit algorithms."""

    def __init__(self, n_arms: int, context_dim: int):
        self.n_arms = n_arms
        self.context_dim = context_dim

    @abstractmethod
    def select_arm(self, context: NDArray[np.float64]) -> int:
        """Select an arm given the current context."""
        pass

    @abstractmethod
    def update(self, context: NDArray[np.float64], arm: int, reward: float) -> None:
        """Update the model with observed reward."""
        pass

    @abstractmethod
    def get_arm_values(self, context: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get expected values for all arms given context."""
        pass
```

```python
# src/bandit/thompson.py
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from .base import BaseBandit
from ..models.bayesian_ridge import BayesianRidgeModel

class ThompsonSamplingBandit(BaseBandit):
    """
    Contextual bandit using Thompson Sampling with Bayesian linear regression.

    Each arm has its own Bayesian Ridge model that estimates:
    - Expected reward given context
    - Uncertainty in that estimate

    Thompson Sampling samples from the posterior to balance exploration/exploitation.
    """

    def __init__(self, n_arms: int, context_dim: int, alpha_init: float = 1.0):
        super().__init__(n_arms, context_dim)
        self.models: List[BayesianRidgeModel] = [
            BayesianRidgeModel(alpha_init=alpha_init)
            for _ in range(n_arms)
        ]
        self.arm_counts = np.zeros(n_arms)

    def select_arm(self, context: NDArray[np.float64]) -> int:
        """Select arm using Thompson Sampling."""
        # TODO: Implement Thompson Sampling
        # 1. For each arm, sample from posterior
        # 2. Return arm with highest sampled value
        pass

    def update(self, context: NDArray[np.float64], arm: int, reward: float) -> None:
        """Update the Bayesian model for the selected arm."""
        # TODO: Update the model for the selected arm
        pass

    def get_arm_values(self, context: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get posterior mean estimates for all arms."""
        # TODO: Return mean predictions from each arm's model
        pass
```

```python
# src/models/bayesian_ridge.py
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import BayesianRidge
from typing import Tuple, Optional

class BayesianRidgeModel:
    """
    Wrapper around sklearn BayesianRidge for bandit reward modeling.

    Provides:
    - predict_with_uncertainty: Returns mean and std for Thompson Sampling
    - sample_prediction: Draws from posterior for arm selection
    """

    def __init__(self, alpha_init: float = 1.0, lambda_init: float = 1.0):
        self.model = BayesianRidge(
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=True
        )
        self.is_fitted = False
        self.X_buffer: list = []
        self.y_buffer: list = []
        self.min_samples = 2  # Need at least 2 samples to fit

    def add_observation(self, context: NDArray[np.float64], reward: float) -> None:
        """Add a new observation to the buffer and refit if possible."""
        # TODO: Add to buffer, refit model when enough samples
        pass

    def predict_with_uncertainty(
        self, context: NDArray[np.float64]
    ) -> Tuple[float, float]:
        """Return (mean, std) prediction for given context."""
        # TODO: Use model.predict with return_std=True
        pass

    def sample_prediction(self, context: NDArray[np.float64]) -> float:
        """Sample from posterior predictive distribution."""
        # TODO: Sample using mean + std * random_normal
        pass
```

---

### Phase 2: Feature Engineering Pipeline

**Objective**: Build the feature extraction system for users and items.

**Scope**:
- `src/features/__init__.py` - Package init
- `src/features/user_features.py` - User context extraction
- `src/features/item_features.py` - Item attribute encoding
- `src/features/normalizer.py` - Feature normalization
- `tests/test_features.py` - Feature pipeline tests

**Deliverables**:
1. `UserFeatureExtractor` class for user context
2. `ItemFeatureExtractor` class for item attributes
3. `FeatureNormalizer` for consistent scaling
4. Combined feature vector generator

**Verification**:
```bash
pytest tests/test_features.py -v
```

**Technical Challenges**:
- Handling missing features gracefully
- Consistent normalization between training and inference

**Debugging Scenarios**:
- If features have wrong shape: Check one-hot encoding dimensions
- If NaN in features: Add missing value imputation

**Time Estimate**: 2-3 hours (cut: simplify to numeric features only)

**Definition of Done**:
- [ ] `UserFeatureExtractor.extract(user_data)` returns fixed-size vector
- [ ] `ItemFeatureExtractor.extract(item_data)` returns fixed-size vector
- [ ] `FeatureNormalizer` fits on training data and transforms consistently
- [ ] All tests pass

**Code Skeleton**:

```python
# src/features/user_features.py
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, List

class UserFeatureExtractor:
    """
    Extract user context features for the bandit.

    Features include:
    - Demographics (age_bucket, etc.)
    - Behavioral (session_count, avg_session_duration)
    - Temporal (hour_of_day, day_of_week)
    """

    def __init__(self, feature_config: Dict[str, Any]):
        self.feature_config = feature_config
        self.feature_dim: int = 0
        self._compute_feature_dim()

    def _compute_feature_dim(self) -> None:
        """Calculate total feature dimension."""
        # TODO: Sum up dimensions from config
        pass

    def extract(self, user_data: Dict[str, Any]) -> NDArray[np.float64]:
        """Extract feature vector from user data."""
        # TODO: Extract and concatenate features
        pass
```

---

### Phase 3: Cold-Start Handling with CF Embeddings

**Objective**: Implement warm-starting for new users and items.

**Scope**:
- `src/coldstart/__init__.py` - Package init
- `src/coldstart/cf_embeddings.py` - Collaborative filtering embeddings
- `src/coldstart/warmup.py` - Warm-start logic for new entities
- `tests/test_coldstart.py` - Cold-start tests

**Deliverables**:
1. `CFEmbedder` class using TruncatedSVD
2. `WarmStartHandler` for initializing new user/item models
3. Integration with main bandit

**Verification**:
```bash
pytest tests/test_coldstart.py -v
```

**Technical Challenges**:
- Building interaction matrix from sparse data
- Incremental updates when new users/items arrive

**Debugging Scenarios**:
- If embeddings are all zeros: Check interaction matrix sparsity
- If new user has no similar users: Fall back to popularity baseline

**Time Estimate**: 3-4 hours (cut: skip incremental updates, use batch retraining)

**Definition of Done**:
- [ ] `CFEmbedder.fit(interactions)` learns embeddings
- [ ] `CFEmbedder.get_user_embedding(user_id)` returns vector
- [ ] `CFEmbedder.get_item_embedding(item_id)` returns vector
- [ ] New users get warm-started with similar user knowledge
- [ ] All tests pass

**Code Skeleton**:

```python
# src/coldstart/cf_embeddings.py
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Optional
import pandas as pd

class CFEmbedder:
    """
    Collaborative filtering embeddings using matrix factorization.

    Uses TruncatedSVD to decompose user-item interaction matrix
    into low-dimensional user and item embeddings.
    """

    def __init__(self, embedding_dim: int = 32, random_state: int = 42):
        self.embedding_dim = embedding_dim
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=random_state)
        self.user_embeddings: Optional[NDArray[np.float64]] = None
        self.item_embeddings: Optional[NDArray[np.float64]] = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Fit embeddings from interaction data.

        Args:
            interactions: DataFrame with columns [user_id, item_id, rating]
        """
        # TODO: Build interaction matrix and fit SVD
        pass

    def get_user_embedding(self, user_id: str) -> NDArray[np.float64]:
        """Get embedding for a user, or average if new."""
        # TODO: Return embedding or fallback
        pass

    def get_item_embedding(self, item_id: str) -> NDArray[np.float64]:
        """Get embedding for an item, or average if new."""
        # TODO: Return embedding or fallback
        pass
```

---

### Phase 4: Offline Policy Evaluation

**Objective**: Implement inverse propensity scoring for offline evaluation.

**Scope**:
- `src/evaluation/__init__.py` - Package init
- `src/evaluation/ips.py` - Inverse propensity scoring estimator
- `src/evaluation/replay.py` - Replay buffer for hybrid learning
- `src/evaluation/metrics.py` - Regret and performance metrics
- `tests/test_evaluation.py` - Evaluation tests

**Deliverables**:
1. `IPSEstimator` for offline policy evaluation
2. `ReplayBuffer` for storing and sampling experiences
3. Regret tracking utilities

**Verification**:
```bash
pytest tests/test_evaluation.py -v
```

**Technical Challenges**:
- High variance in IPS estimates (need clipping)
- Propensity score estimation for logged data

**Debugging Scenarios**:
- If IPS estimate has huge variance: Apply weight clipping
- If replay buffer runs out of memory: Implement FIFO eviction

**Time Estimate**: 3-4 hours (cut: skip doubly robust estimator)

**Definition of Done**:
- [ ] `IPSEstimator.estimate(policy, logged_data)` returns unbiased estimate
- [ ] `ReplayBuffer.add(experience)` stores experience
- [ ] `ReplayBuffer.sample(batch_size)` returns random batch
- [ ] Variance reduction via clipping works
- [ ] All tests pass

**Code Skeleton**:

```python
# src/evaluation/ips.py
import numpy as np
from numpy.typing import NDArray
from typing import List, Callable
from dataclasses import dataclass

@dataclass
class LoggedInteraction:
    """A single logged interaction for offline evaluation."""
    context: NDArray[np.float64]
    action: int
    reward: float
    propensity: float  # P(action | context) under logging policy

class IPSEstimator:
    """
    Inverse Propensity Scoring estimator for offline policy evaluation.

    Estimates expected reward of a new policy using logged data
    from a different (logging) policy.
    """

    def __init__(self, clip_min: float = 0.01, clip_max: float = 100.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def estimate(
        self,
        target_policy: Callable[[NDArray[np.float64]], int],
        logged_data: List[LoggedInteraction]
    ) -> float:
        """
        Estimate expected reward of target_policy.

        Args:
            target_policy: Function mapping context -> action
            logged_data: List of logged interactions

        Returns:
            Estimated expected reward
        """
        # TODO: Implement IPS with importance weighting
        pass
```

```python
# src/evaluation/replay.py
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from collections import deque
from dataclasses import dataclass
import random

@dataclass
class Experience:
    """A single experience for replay."""
    context: NDArray[np.float64]
    action: int
    reward: float
    timestamp: float

class ReplayBuffer:
    """
    Experience replay buffer for hybrid online/offline learning.

    Stores experiences and allows random sampling for batch updates.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        # TODO: Add to deque
        pass

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        # TODO: Random sample
        pass

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### Phase 5: Interpretable Explanations

**Objective**: Generate human-readable explanations for recommendations.

**Scope**:
- `src/explanations/__init__.py` - Package init
- `src/explanations/feature_contrib.py` - Feature contribution analysis
- `src/explanations/generator.py` - Natural language explanation generation
- `tests/test_explanations.py` - Explanation tests

**Deliverables**:
1. `FeatureContributionAnalyzer` for computing feature importance
2. `ExplanationGenerator` for producing readable explanations
3. Uncertainty communication in explanations

**Verification**:
```bash
pytest tests/test_explanations.py -v
```

**Technical Challenges**:
- Linear models make this easier (coefficients = importance)
- Communicating uncertainty without confusing users

**Time Estimate**: 2-3 hours (cut: skip uncertainty visualization)

**Definition of Done**:
- [ ] `FeatureContributionAnalyzer.analyze(context, arm)` returns importance dict
- [ ] `ExplanationGenerator.explain(context, arm, contributions)` returns string
- [ ] Explanations are human-readable and accurate
- [ ] All tests pass

**Code Skeleton**:

```python
# src/explanations/feature_contrib.py
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List

class FeatureContributionAnalyzer:
    """
    Analyze feature contributions to recommendations.

    For linear models, contribution = coefficient * feature_value.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def analyze(
        self,
        context: NDArray[np.float64],
        model_coefficients: NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Compute contribution of each feature.

        Returns:
            Dict mapping feature_name -> contribution
        """
        # TODO: Compute and return contributions
        pass
```

```python
# src/explanations/generator.py
from typing import Dict

class ExplanationGenerator:
    """Generate human-readable explanations for recommendations."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def explain(
        self,
        item_name: str,
        contributions: Dict[str, float],
        uncertainty: float
    ) -> str:
        """
        Generate explanation string.

        Example output:
        "We recommend 'Article X' because:
         - Your interest in technology (+0.42)
         - Morning browsing pattern (+0.31)
         - Similar users liked it (+0.28)
         Confidence: High (uncertainty: 0.12)"
        """
        # TODO: Generate explanation string
        pass
```

---

### Phase 6: API Layer and Integration

**Objective**: Build FastAPI service integrating all components.

**Scope**:
- `src/api/__init__.py` - Package init
- `src/api/main.py` - FastAPI application
- `src/api/schemas.py` - Pydantic request/response models
- `src/api/engine.py` - RecommendationEngine orchestrator
- `tests/test_api.py` - API integration tests

**Deliverables**:
1. FastAPI application with endpoints
2. RecommendationEngine tying all components together
3. Health check and monitoring endpoints

**Verification**:
```bash
pytest tests/test_api.py -v
uvicorn src.api.main:app --reload  # Manual test
```

**Time Estimate**: 3-4 hours (cut: skip async, use sync for simplicity)

**Definition of Done**:
- [ ] `POST /recommend` returns recommendation with explanation
- [ ] `POST /feedback` records reward and updates model
- [ ] `GET /health` returns system status
- [ ] API docs auto-generated at `/docs`
- [ ] All tests pass

**Code Skeleton**:

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from .schemas import RecommendRequest, RecommendResponse, FeedbackRequest
from .engine import RecommendationEngine

app = FastAPI(
    title="Contextual Bandit Recommender",
    description="Content recommendation using Thompson Sampling",
    version="0.1.0"
)

engine = RecommendationEngine()

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendation(request: RecommendRequest):
    """Get a personalized recommendation for a user."""
    # TODO: Implement
    pass

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Submit reward feedback for a recommendation."""
    # TODO: Implement
    pass

@app.get("/health")
def health_check():
    """Check system health."""
    return {"status": "healthy", "model_loaded": engine.is_ready()}
```

```python
# src/api/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class RecommendRequest(BaseModel):
    user_id: str
    user_context: Dict[str, Any]
    n_recommendations: int = 1

class RecommendResponse(BaseModel):
    recommendations: List[str]
    explanations: List[str]
    confidence_scores: List[float]

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    reward: float  # 0.0 to 1.0
```

---

## Risk Assessment

| Risk | Likelihood | Impact | L×I | Early Warning | Mitigation |
|------|------------|--------|-----|---------------|------------|
| Numerical instability in Bayesian updates | Medium | High | 🔴 | NaN/Inf in predictions | Add regularization, use stable solvers |
| Cold-start embeddings fail to generalize | Medium | Medium | 🟡 | New user performance drops | Fallback to popularity baseline |
| IPS variance too high for reliable eval | High | Medium | 🟡 | Wild estimate swings | Aggressive clipping, larger samples |
| Feature engineering doesn't capture signal | Medium | High | 🔴 | Regret doesn't decrease | A/B test feature sets, add interactions |
| Model updates too slow for real-time | Low | Medium | 🟢 | API latency > 100ms | Batch updates, async processing |
| Overfitting on small datasets | High | Medium | 🟡 | Train/test gap grows | Regularization, cross-validation |

---

## Testing Strategy

### Testing Pyramid

```
                    ┌─────────────┐
                    │   System    │  ← 1-2 end-to-end tests
                    │   Tests     │
                    ├─────────────┤
                    │ Integration │  ← 5-10 component integration tests
                    │   Tests     │
                    ├─────────────┤
                    │    Unit     │  ← 30+ unit tests
                    │   Tests     │
                    └─────────────┘
```

### Framework: pytest

### First Three Tests to Write

```python
# tests/test_bandit_basic.py

def test_bandit_initialization():
    """Bandit initializes with correct dimensions."""
    bandit = ThompsonSamplingBandit(n_arms=10, context_dim=5)
    assert bandit.n_arms == 10
    assert bandit.context_dim == 5
    assert len(bandit.models) == 10

def test_bandit_arm_selection_returns_valid_arm():
    """select_arm returns integer in valid range."""
    bandit = ThompsonSamplingBandit(n_arms=10, context_dim=5)
    context = np.random.randn(5)
    arm = bandit.select_arm(context)
    assert isinstance(arm, int)
    assert 0 <= arm < 10

def test_bandit_update_changes_model():
    """update() modifies the model state."""
    bandit = ThompsonSamplingBandit(n_arms=10, context_dim=5)
    context = np.random.randn(5)
    arm = 0

    # Get prediction before
    pred_before = bandit.models[arm].predict_with_uncertainty(context)

    # Update with reward
    bandit.update(context, arm, reward=1.0)
    bandit.update(context, arm, reward=1.0)  # Need 2+ for fitting

    # Model should now be fitted
    assert bandit.models[arm].is_fitted
```

---

## First Concrete Task

**File to Create**: `src/bandit/base.py`

**First Function Signature**:
```python
def select_arm(self, context: NDArray[np.float64]) -> int:
```

**Starter Code** (copy-paste ready):

```python
# src/bandit/base.py
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
```

**Verification Method**:
```bash
cd /Users/sakeeb/Code\ repositories/contextual-bandit-recommender
python -m src.bandit.base
# Should print: "OK: Cannot instantiate abstract class - ..."
```

**First Commit Message**:
```
Add BaseBandit abstract class with core interface

Define the abstract interface for contextual bandits with:
- select_arm(): Choose action given context
- update(): Learn from observed rewards
- get_arm_values(): Get expected values for exploration
```

---

## Project Directory Structure

```
contextual-bandit-recommender/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── src/
│   ├── __init__.py
│   ├── bandit/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── thompson.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── bayesian_ridge.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── user_features.py
│   │   ├── item_features.py
│   │   └── normalizer.py
│   ├── coldstart/
│   │   ├── __init__.py
│   │   ├── cf_embeddings.py
│   │   └── warmup.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ips.py
│   │   ├── replay.py
│   │   └── metrics.py
│   ├── explanations/
│   │   ├── __init__.py
│   │   ├── feature_contrib.py
│   │   └── generator.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       └── engine.py
├── tests/
│   ├── __init__.py
│   ├── test_bandit_basic.py
│   ├── test_features.py
│   ├── test_coldstart.py
│   ├── test_evaluation.py
│   ├── test_explanations.py
│   └── test_api.py
├── data/
│   └── sample/
│       └── .gitkeep
└── notebooks/
    └── exploration.ipynb
```
