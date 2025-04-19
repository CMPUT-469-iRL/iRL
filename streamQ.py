
from __future__ import annotations

import os
import random
from collections import deque
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tmaze_pen import TMazeEnv
from eLSTM_model.model import RTRLQuasiLSTMModel


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _init_hidden(model: RTRLQuasiLSTMModel, batch: int, device: torch.device):
    """Create a fresh hidden state for `batch` sequences on *device*."""
    return model.get_init_states(batch, device)


class StreamQAgent:
    """Deep‑Q agent that uses an RTRL‑trained Quasi‑LSTM as its Q‑network."""

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10_000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        no_embedding: bool = False,
    ) -> None:
        # Networks ------------------------------------------------------
        self.q_net = RTRLQuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding,
        )
        self.tgt_net = RTRLQuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding,
        )
        self.update_target()

        # Device --------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.tgt_net.to(self.device)

        # Optim / loss --------------------------------------------------
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.crit = nn.MSELoss()

        # Hyper‑parameters ---------------------------------------------
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.batch_size = batch_size
        self.tgt_freq = target_update_freq
        self.out_dim = output_size

        # Replay memory -------------------------------------------------
        self.mem: deque[Tuple[int, int, float, int, bool]] = deque(maxlen=memory_size)

        # Hidden state --------------------------------------------------
        self.h = _init_hidden(self.q_net, 1, self.device)

    # ------------------------------------------------------------------
    def reset_hidden(self, B: int = 1):
        self.h = _init_hidden(self.q_net, B, self.device)

    @torch.no_grad()
    def act(self, s: int, *, train: bool = True) -> int:
        if train and random.random() < self.epsilon:
            return random.randrange(self.out_dim)
        s_t = torch.tensor([s], dtype=torch.long, device=self.device)  # (1,)
        q, _, self.h = self.q_net(s_t, self.h)
        return q.squeeze(0).argmax().item()

    # ------------------------------------------------------------------
    def store(self, s, a, r, s2, d):
        self.mem.append((s, a, r, s2, d))

    def _sample(self):
        batch = random.sample(self.mem, self.batch_size)
        return map(list, zip(*batch))  # states, actions, ...

    def step(self) -> float:
        if len(self.mem) < self.batch_size:
            return 0.0
        S, A, R, S2, D = self._sample()
        S_t  = torch.tensor(S,  dtype=torch.long,    device=self.device)
        A_t  = torch.tensor(A,  dtype=torch.long,    device=self.device).unsqueeze(-1)
        R_t  = torch.tensor(R,  dtype=torch.float32, device=self.device).unsqueeze(-1)
        S2_t = torch.tensor(S2, dtype=torch.long,    device=self.device)
        D_t  = torch.tensor(D,  dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.opt.zero_grad()
        h0 = _init_hidden(self.q_net, self.batch_size, self.device)
        q, _, _ = self.q_net(S_t, h0)
        q = q.gather(-1, A_t)

        with torch.no_grad():
            ht = _init_hidden(self.tgt_net, self.batch_size, self.device)
            q2, _, _ = self.tgt_net(S2_t, ht)
            target = R_t + (1 - D_t) * self.gamma * q2.max(-1, keepdim=True)[0]

        loss = self.crit(q, target)
        loss.backward(); self.opt.step();
        return loss.item()

    # ------------------------------------------------------------------
    def decay_eps(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target(self):
        self.tgt_net.load_state_dict(self.q_net.state_dict())


# ---------------------------------------------------------------------
# Evaluation & plotting helpers
# ---------------------------------------------------------------------

def _eval(agent: StreamQAgent, env: TMazeEnv, ep: int = 10, max_steps: int = 100):
    tot = 0.0
    for _ in range(ep):
        s, done, steps = env.reset(), False, 0
        agent.reset_hidden()
        while not done and steps < max_steps:
            a = agent.act(s, train=False)
            s, r, done, _ = env.step(a)
            tot += r; steps += 1
    return tot / ep


def _plot(rew: List[float], loss: List[float], eval_r: List[float], k: int):
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.plot(rew); plt.title("R per episode"); plt.xlabel("episode");
    plt.subplot(132); plt.plot(loss); plt.title("loss"); plt.xlabel("episode")
    plt.subplot(133); plt.plot(range(k, len(rew)+1, k), eval_r); plt.title("eval R̄");
    plt.tight_layout(); plt.savefig("streamQ_training.png"); plt.close()


# ---------------------------------------------------------------------
# Public API – called by external driver scripts
# ---------------------------------------------------------------------

def train_stream_q(
    *,
    episodes: int = 500,
    corridor_length: int = 5,
    hidden_size: int = 64,
    embedding_size: int = 32,
    lr: float = 1e-3,
    gamma: float = 0.99,
    eval_interval: int = 20,
    batch_size: int = 32,
    target_update_freq: int = 10,
    max_steps_per_episode: int = 100,
    max_time_steps: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Train Stream‑Q agent and (optionally) save a checkpoint to *save_path*."""
    env = TMazeEnv(corridor_length=corridor_length)
    agent = StreamQAgent(
        input_size=4,
        output_size=3,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )

    R_hist, L_hist, E_hist = [], [], []
    global_steps = 0

    for ep in range(episodes):
        s = env.reset(); agent.reset_hidden(); done, ep_r, ep_l, steps = False, 0.0, 0.0, 0
        while not done and steps < max_steps_per_episode:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.store(s, a, r, s2, done)
            ep_l += agent.step(); s, ep_r, steps = s2, ep_r + r, steps + 1
            global_steps += 1
            if max_time_steps and global_steps >= max_time_steps:
                print(f"Reached {max_time_steps} steps -> early stop")
                return agent, R_hist
        # post‑episode
        agent.decay_eps();
        if ep % agent.tgt_freq == 0: agent.update_target()
        R_hist.append(ep_r); L_hist.append(ep_l / max(1, steps))
        if (ep + 1) % eval_interval == 0:
            E_hist.append(_eval(agent, env, 10, max_steps_per_episode))
            print(f"Ep{ep+1}: R={ep_r:.2f} meanR10={np.mean(R_hist[-10:]):.2f} eps={agent.epsilon:.3f} eval={E_hist[-1]:.2f}")

    _plot(R_hist, L_hist, E_hist, eval_interval)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model": agent.q_net.state_dict(),
            "opt": agent.opt.state_dict(),
            "R_hist": R_hist,
            "L_hist": L_hist,
            "E_hist": E_hist,
            "steps": global_steps,
            "corridor": corridor_length,
            "hidden_size": hidden_size,
        }, save_path)
        print(f"Checkpoint written to {save_path}")

    return agent, R_hist


# The module exposes *only* the public training function ----------------
__all__ = ["train_stream_q"]
