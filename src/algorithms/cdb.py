from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


class ComparisonRecord:
    __slots__ = ("action_features", "a1_idx", "a2_idx", "y_tilde")

    def __init__(
        self,
        action_features: np.ndarray,
        a1_idx: int,
        a2_idx: int,
        y_tilde: float,
    ) -> None:
        self.action_features = action_features
        self.a1_idx = a1_idx
        self.a2_idx = a2_idx
        self.y_tilde = y_tilde


class EmbeddingFeatureExtractor:
    def __init__(
        self,
        embed_layer: torch.nn.Embedding,
        tokenizer,
        p: int = 128,
        seed: int = 0,
    ) -> None:
        self.embed_layer = embed_layer
        self.tokenizer = tokenizer
        self.p = p
        self.d_model = embed_layer.embedding_dim
        self.device = embed_layer.weight.device

        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(p)
        self.W_ctx = rng.randn(p, self.d_model).astype(np.float32) * scale
        self.W_act = rng.randn(p, self.d_model).astype(np.float32) * scale
        self._act_cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _mean_embed(self, text: str, max_tokens: int = 512) -> np.ndarray:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            return np.zeros(self.d_model, dtype=np.float32)
        ids = ids[-max_tokens:]
        t_ids = torch.tensor([ids], device=self.device)
        emb = self.embed_layer(t_ids)
        return emb[0].mean(dim=0).cpu().float().numpy()

    def encode_context(self, prompt: str) -> np.ndarray:
        raw = self._mean_embed(prompt)
        return (self.W_ctx @ raw).astype(np.float64)

    def _action_proj(self, action: str) -> np.ndarray:
        if action not in self._act_cache:
            raw = self._mean_embed(action, max_tokens=64)
            self._act_cache[action] = (self.W_act @ raw).astype(np.float64)
        return self._act_cache[action]

    def feature(self, context: np.ndarray, action: str) -> np.ndarray:
        return context * self._action_proj(action)

    def all_action_features(
        self,
        context: np.ndarray,
        actions: Sequence[str],
    ) -> np.ndarray:
        return np.stack([self.feature(context, a) for a in actions])

    def clear_cache(self) -> None:
        self._act_cache.clear()


class FGTSCDB:
    def __init__(
        self,
        p: int = 128,
        eta: float = 1.0,
        mu: float = 0.1,
        delta: float = 0.01,
        sigma0: float = 1.0,
        max_history: int = 2000,
    ) -> None:
        self.p = p
        self.eta = eta
        self.mu = mu
        self.delta = delta
        self.sigma0 = sigma0
        self.max_history = max_history
        self.theta1 = np.random.randn(p).astype(np.float64) * 0.01
        self.theta2 = np.random.randn(p).astype(np.float64) * 0.01
        self.history: List[ComparisonRecord] = []

    def sgld_step(self) -> None:
        g1 = self._grad_I(j=1)
        g2 = self._grad_I(j=2)
        s = np.sqrt(2.0 * self.delta)
        self.theta1 = self.theta1 - self.delta * g1 + s * np.random.randn(self.p)
        self.theta2 = self.theta2 - self.delta * g2 + s * np.random.randn(self.p)

    def select_actions(self, action_features: np.ndarray) -> Tuple[int, int]:
        s1 = action_features @ self.theta1
        s2 = action_features @ self.theta2
        a1 = int(np.argmax(s1))
        a2 = int(np.argmax(s2))
        if a1 == a2:
            alt = s2.copy()
            alt[a1] = -np.inf
            a2 = int(np.argmax(alt))
        return a1, a2

    def record(
        self,
        action_features: np.ndarray,
        a1_idx: int,
        a2_idx: int,
        y_tilde: float,
    ) -> None:
        self.history.append(
            ComparisonRecord(action_features.copy(), a1_idx, a2_idx, y_tilde)
        )
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def uncertainty(self, action_features: np.ndarray) -> float:
        s1 = action_features @ self.theta1
        s2 = action_features @ self.theta2
        a1 = int(np.argmax(s1))
        a2 = int(np.argmax(s2))
        if a1 == a2:
            alt = s2.copy()
            alt[a1] = -np.inf
            a2 = int(np.argmax(alt))
        theta_avg = (self.theta1 + self.theta2) / 2.0
        gap = abs(float(np.dot(theta_avg, action_features[a1] - action_features[a2])))
        return -gap

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))

    def _grad_I(self, j: int) -> np.ndarray:
        theta = self.theta1 if j == 1 else self.theta2
        grad = theta / (self.sigma0 ** 2)
        for rec in self.history:
            psi1 = rec.action_features[rec.a1_idx]
            psi2 = rec.action_features[rec.a2_idx]
            dpsi = psi1 - psi2
            z = rec.y_tilde * float(np.dot(theta, dpsi))
            sig = self._sigmoid(z)
            grad += self.eta * sig * (1.0 - sig) * rec.y_tilde * dpsi
            scores = rec.action_features @ theta
            a_star = int(np.argmax(scores))
            ref_idx = rec.a2_idx if j == 1 else rec.a1_idx
            grad -= self.mu * (rec.action_features[a_star] - rec.action_features[ref_idx])
        return grad
