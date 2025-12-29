# Contextual Bandit Recommender

A content recommendation system using contextual multi-armed bandits to balance exploration and exploitation. The system learns user preferences while minimizing regret, handles cold-start scenarios for new users and items, and provides interpretable explanations for its recommendations.

## Overview

This project implements a production-ready contextual bandit system for content personalization. It combines Thompson Sampling with linear payoff models (LinUCB-style), Bayesian ridge regression for uncertainty quantification, and collaborative filtering embeddings for warm-starting new users and items.

## Key Features

- **Thompson Sampling with Linear Payoffs**: Implements LinUCB-style contextual bandits for personalized recommendations
- **Bayesian Uncertainty Quantification**: Uses Bayesian ridge regression to model reward uncertainty and drive exploration
- **Cold-Start Handling**: Warm-starts new users/items using collaborative filtering embeddings
- **Offline Policy Evaluation**: Evaluates policies using inverse propensity scoring (IPS) before deployment
- **Hybrid Learning**: Combines online learning with offline replay buffer for sample-efficient updates
- **Interpretable Recommendations**: Provides human-readable explanations for why items are recommended

## Architecture

```
User Request → Feature Engineering → Bandit Policy → Item Selection → Feedback Loop
                    ↓                      ↓               ↓
              User Context          Bayesian Model    Reward Signal
              Item Features         Uncertainty       Policy Update
              CF Embeddings         Exploration
```

## Requirements

- Python 3.9+
- NumPy, SciPy (numerical computing)
- scikit-learn (Bayesian ridge regression, embeddings)
- pandas (data handling)
- FastAPI (API serving)
- pytest (testing)

## Installation

```bash
git clone https://github.com/Sakeeb91/contextual-bandit-recommender.git
cd contextual-bandit-recommender
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from bandit import ContextualBandit
from features import FeatureEngineering

# Initialize the bandit system
bandit = ContextualBandit(n_arms=100, context_dim=64)

# Get recommendation for a user
user_context = feature_eng.get_user_context(user_id="user_123")
recommended_item, explanation = bandit.recommend(user_context)

# Update with feedback
bandit.update(user_context, recommended_item, reward=1.0)
```

## Project Structure

```
contextual-bandit-recommender/
├── src/
│   ├── bandit/           # Core bandit algorithms
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Bayesian reward models
│   ├── evaluation/       # Offline policy evaluation
│   └── api/              # FastAPI serving layer
├── tests/                # Unit and integration tests
├── docs/                 # Documentation and implementation plan
├── data/                 # Sample datasets
└── notebooks/            # Exploration and analysis
```

## Documentation

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the detailed implementation roadmap and technical specifications.

## License

MIT License

## Author

Sakeeb Rahman (rahman.sakeeb@gmail.com)
