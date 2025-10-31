"""Reinforcement Learning agents for travel recommendations."""

import numpy as np


class EpsilonGreedyAgent:
    """Epsilon-Greedy Multi-Armed Bandit Agent."""
    
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, k=3):
        """Select arm using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms, size=k, replace=False)
        else:
            return np.argsort(self.values)[::-1][:k]

    def update(self, arm, reward):
        """Update arm's estimated value."""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class LinUCBAgent:
    """Linear Upper Confidence Bound Contextual Bandit Agent."""
    
    def __init__(self, n_arms, n_features, alpha=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, x, k=3):
        """Select arm with highest UCB score given context."""
        p = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            p[i] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        ranked = np.argsort(p)[::-1][:k]
        return ranked, p

    def update(self, arm, x, reward):
        """Update ridge regression parameters for selected arm."""
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class ContextualThompsonSampling:
    """Contextual Thompson Sampling Agent using Bayesian linear regression."""
    
    def __init__(self, n_arms, d, alpha=0.1):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.B = [np.identity(d) for _ in range(n_arms)]
        self.mu = [np.zeros(d) for _ in range(n_arms)]
        self.f = [np.zeros(d) for _ in range(n_arms)]

    def select_arm(self, x, k=3):
        """Select arm using Thompson sampling."""
        sampled_theta = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            B_inv = np.linalg.inv(self.B[i])
            mu_hat = B_inv @ self.f[i]
            theta_sample = np.random.multivariate_normal(mu_hat, self.alpha**2 * B_inv)
            sampled_theta[i] = theta_sample @ x
        ranked = np.argsort(sampled_theta)[::-1][:k]
        return ranked, sampled_theta 

    def update(self, arm, x, reward):
        """Update Bayesian parameters for selected arm."""
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
