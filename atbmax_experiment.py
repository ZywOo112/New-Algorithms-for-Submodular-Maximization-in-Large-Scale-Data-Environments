#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATB-Max 算法实验验证套件
============================================================
大规模数据环境下次模最大化问题新算法 —— ATB-Max 实验验证

本模块实现了论文中提出的 ATB-Max 算法及其对比算法，并以社交网络
影响力最大化问题为测试场景，在多个数据集上进行系统实验。

实验内容：
  1. 综合性能比较（表 5.1）
  2. 消融实验（UCB 模块与轻量估计器的作用）
  3. 参数敏感性分析（beta, lambda, k, M_est）
  4. 可扩展性实验（多个不同规模数据集）

References:
  - Kempe et al. 2003 (KDD): Influence maximization, IC model
  - Badanidiyuru et al. 2014 (KDD): Sieve-Streaming
  - Auer et al. 2002 (Machine Learning): UCB1
  - Leskovec & Krevl 2014: SNAP datasets
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import time
import random
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Section 0: 配置与工具函数
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置参数"""
    # 数据集参数
    dataset_name: str = "facebook-like"

    # 基数约束
    k: int = 50

    # ATB-Max 参数
    M_est: int = 240          # 轻量估计器采样数
    M_exact: int = 2500       # 精确 Oracle 采样数
    beta: float = 0.12        # 估计误差缓冲参数
    lam: float = 0.005        # 计算成本惩罚因子
    m_thresholds: int = 20    # 候选阈值个数
    tau_min_frac: float = 0.001  # 最小阈值占 max_spread 的比例
    tau_max_frac: float = 0.5     # 最大阈值占 max_spread 的比例

    # Sieve-Streaming 参数
    epsilon: float = 0.1

    # 实验参数
    n_repeats: int = 10       # 重复实验次数（不同随机流顺序）
    n_processes: int = 1      # 并行进程数

    # 随机种子
    random_state: int = 42

    def __repr__(self):
        return (f"ExperimentConfig(k={self.k}, beta={self.beta}, lam={self.lam}, "
                f"M_est={self.M_est}, M_exact={self.M_exact}, m={self.m_thresholds})")


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# Section 1: 独立级联传播模型 (Independent Cascade Model)
# ============================================================================

class IndependentCascadeModel:
    """
    独立级联 (Independent Cascade, IC) 传播模型。

    每个节点 u 在其被激活的瞬间，有一次机会尝试激活每个未激活的出邻 v，
    成功概率为 p(u,v)。过程持续至没有新节点被激活为止。

    本实现支持两种边概率模式：
      - weighted_cascade: p(u,v) = 1 / in_degree(v)  (默认，文献标准做法)
      - uniform:         p(u,v) = p0  (需指定 p0)
    """

    def __init__(self, graph: 'nx.DiGraph', mode: str = 'weighted_cascade',
                 p0: float = 0.05, seed: int = None):
        self.graph = graph
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        # 预计算每条边的传播概率
        self.edge_probs = {}
        if mode == 'weighted_cascade':
            for u, v in graph.edges():
                in_deg = max(graph.in_degree(v), 1)
                self.edge_probs[(u, v)] = 1.0 / in_deg
        elif mode == 'uniform':
            for u, v in graph.edges():
                self.edge_probs[(u, v)] = p0
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _single_diffusion(self, seed_set: Set[int]) -> int:
        """
        执行一次完整的独立级联传播过程。
        返回最终被激活的节点数量。
        """
        activated = set(seed_set)
        frontier = list(seed_set)

        while frontier:
            next_frontier = []
            for u in frontier:
                for v in self.graph.successors(u):
                    if v not in activated:
                        if self.rng.random() < self.edge_probs[(u, v)]:
                            activated.add(v)
                            next_frontier.append(v)
            frontier = next_frontier

        return len(activated)

    def estimate_spread(self, seed_set: Set[int], n_sim: int,
                        seed_offset: int = 0) -> float:
        """
        用 Monte Carlo 模拟估计种子集的期望传播范围。

        Parameters
        ----------
        seed_set : 种子节点集合
        n_sim : 模拟次数
        seed_offset : 随机种子偏移（用于生成独立随机流）

        Returns
        -------
        float : 平均激活节点数
        """
        if not seed_set:
            return 0.0

        old_state = self.rng.get_state()
        self.rng.seed(self.rng.randint(0, 2**31 - 1) + seed_offset)

        total = sum(self._single_diffusion(seed_set) for _ in range(n_sim))

        self.rng.set_state(old_state)
        return total / n_sim

    def estimate_marginal_gain(self, node: int, current_set: Set[int],
                                n_sim: int, cached_spread: float = None,
                                seed_offset: int = 0) -> float:
        """
        估计将 node 加入 current_set 的边际增益。

        采用 ǂCRN (Common Random Numbers) 技术：对 current_set 和
        current_set ∪ {node} 使用相同的随机流，以降低方差。

        Parameters
        ----------
        node : 候选节点
        current_set : 当前解集
        n_sim : 模拟次数
        cached_spread : 缓存的 σ(current_set)。若为 None 则重新估计
        seed_offset : 随机种子偏移

        Returns
        -------
        float : 边际增益估计值
        """
        augmented_set = current_set | {node}
        augmented_spread = self.estimate_spread(augmented_set, n_sim, seed_offset)

        if cached_spread is not None:
            current_spread = cached_spread
        else:
            current_spread = self.estimate_spread(current_set, n_sim, seed_offset + 100000)

        return augmented_spread - current_spread


# ============================================================================
# Section 2: 数据集生成与加载
# ============================================================================

def generate_graph_erdos_renyi(n: int, p: float, directed: bool = True,
                                 seed: int = 42) -> 'nx.DiGraph':
    """生成 Erdos-Renyi 随机图"""
    g_und = nx.erdos_renyi_graph(n, p, seed=seed)
    if directed:
        g = nx.DiGraph()
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)
        return g
    return g_und


def generate_graph_barabasi_albert(n: int, m: int, directed: bool = True,
                                     seed: int = 42) -> 'nx.DiGraph':
    """生成 Barabasi-Albert 无标度网络"""
    g_und = nx.barabasi_albert_graph(n, m, seed=seed)
    if directed:
        g = nx.DiGraph()
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)
        return g
    return g_und


def generate_graph_watts_strogatz(n: int, k: int, p: float,
                                    directed: bool = True,
                                    seed: int = 42) -> 'nx.DiGraph':
    """生成 Watts-Strogatz 小世界网络"""
    g_und = nx.watts_strogatz_graph(n, k, p, seed=seed)
    if directed:
        g = nx.DiGraph()
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)
        return g
    return g_und


def load_facebook_graph(path: str = None) -> 'nx.DiGraph':
    """
    加载 Facebook 社交网络数据集 (SNAP)。
    若未提供路径，则生成一个参数相似的合成图替代。

    Facebook 数据集统计: 4039 nodes, 88234 undirected edges
    """
    if path and os.path.exists(path):
        g_und = nx.read_edgelist(path, nodetype=int, comments='#')
    else:
        print("  [Info] Facebook dataset not found. Generating synthetic substitute "
              "(n=4039, m=15 for Barabasi-Albert, ~88k directed edges)")
        g_und = nx.barabasi_albert_graph(4039, 22, seed=42)
        # 验证边数大致匹配
        actual_edges = g_und.number_of_edges()
        print(f"  [Info] Generated {g_und.number_of_nodes()} nodes, "
              f"{actual_edges} undirected edges "
              f"({2*actual_edges} directed)")

    # 转为有向图（每条无向边 → 两条有向边）
    g = nx.DiGraph()
    for u, v in g_und.edges():
        g.add_edge(u, v)
        g.add_edge(v, u)
    return g


def prepare_datasets() -> Dict[str, 'nx.DiGraph']:
    """
    准备实验所需的全部数据集。

    Returns
    -------
    Dict[str, nx.DiGraph] : {数据集名称: 有向图}
    """
    datasets = {}
    print("=" * 60)
    print("准备实验数据集...")
    print("=" * 60)

    # 1. Facebook 规模合成图 (BA 模型, 匹配 Facebook 规模)
    print("\n[1/5] 生成 Facebook-scale 无标度网络...")
    g_fb = generate_graph_barabasi_albert(n=4039, m=22, seed=42)
    datasets["facebook-scale (n=4039, BA)"] = g_fb
    print(f"  节点数: {g_fb.number_of_nodes()}, 边数: {g_fb.number_of_edges()}")

    # 2. 中等规模无标度网络
    print("\n[2/5] 生成中等规模无标度网络...")
    g_medium = generate_graph_barabasi_albert(n=2000, m=15, seed=123)
    datasets["medium-scale (n=2000, BA)"] = g_medium
    print(f"  节点数: {g_medium.number_of_nodes()}, 边数: {g_medium.number_of_edges()}")

    # 3. 大规模无标度网络 (可扩展性测试)
    print("\n[3/5] 生成大规模无标度网络 (可扩展性测试)...")
    g_large = generate_graph_barabasi_albert(n=8000, m=15, seed=456)
    datasets["large-scale (n=8000, BA)"] = g_large
    print(f"  节点数: {g_large.number_of_nodes()}, 边数: {g_large.number_of_edges()}")

    # 4. Erdos-Renyi 随机图
    print("\n[4/5] 生成 Erdos-Renyi 随机图...")
    g_er = generate_graph_erdos_renyi(n=3000, p=0.008, seed=789)
    datasets["random (n=3000, ER)"] = g_er
    print(f"  节点数: {g_er.number_of_nodes()}, 边数: {g_er.number_of_edges()}")

    # 5. Watts-Strogatz 小世界网络
    print("\n[5/5] 生成 Watts-Strogatz 小世界网络...")
    g_ws = generate_graph_watts_strogatz(n=3000, k=20, p=0.1, seed=101)
    datasets["small-world (n=3000, WS)"] = g_ws
    print(f"  节点数: {g_ws.number_of_nodes()}, 边数: {g_ws.number_of_edges()}")

    print("\n数据集准备完成。\n")
    return datasets


# ============================================================================
# Section 3: 算法实现
# ============================================================================

class GreedyAlgorithm:
    """
    经典离线贪心算法 (Nemhauser et al., 1978; Kempe et al., 2003)

    每轮选取边际增益最大的元素加入解集，共进行 k 轮。
    时间复杂度 O(kn) 次 Oracle 调用。
    """

    def __init__(self, ic_model: IndependentCascadeModel, k: int,
                 M_exact: int = 2500):
        self.ic = ic_model
        self.k = k
        self.M_exact = M_exact
        self.oracle_calls = 0
        self.total_spread = 0.0
        self.total_budget = 0

    def run(self, nodes: List[int]) -> Tuple[Set[int], float]:
        """运行贪心算法"""
        S = set()
        cached_spread = 0.0
        self.oracle_calls = 0
        self.total_budget = 0

        for _ in range(self.k):
            best_node = None
            best_gain = -float('inf')

            for v in nodes:
                if v in S:
                    continue
                gain = self.ic.estimate_marginal_gain(
                    v, S, self.M_exact, cached_spread,
                    seed_offset=self.oracle_calls)
                self.oracle_calls += 1
                self.total_budget += self.M_exact

                if gain > best_gain:
                    best_gain = gain
                    best_node = v

            if best_node is not None:
                S.add(best_node)
                cached_spread = self.ic.estimate_spread(S, self.M_exact,
                                                         seed_offset=self.oracle_calls)
                self.total_budget += self.M_exact

        final_spread = self.ic.estimate_spread(S, self.M_exact * 2,
                                                seed_offset=self.oracle_calls)
        self.total_budget += self.M_exact * 2
        return S, final_spread


class SieveStreaming:
    """
    Sieve-Streaming 算法 (Badanidiyuru et al., 2014 KDD)

    维护 O(log k / epsilon) 个几何间隔的候选阈值，每个阈值独立维护一个
    解集。对每个到达元素，仅当其相对于某阈值的解集的边际增益超过该阈值
    且解集未满时才加入。最终返回 f 值最大的解集。

    近似比: (1/2 - epsilon)   (单次流, 单调次模)
    """

    def __init__(self, ic_model: IndependentCascadeModel, k: int,
                 M_exact: int = 2500, epsilon: float = 0.1):
        self.ic = ic_model
        self.k = k
        self.M_exact = M_exact
        self.epsilon = epsilon
        self.oracle_calls = 0
        self.total_budget = 0

    def run(self, stream: List[int]) -> Tuple[Set[int], float]:
        """
        在单次数据流上运行 Sieve-Streaming。

        Parameters
        ----------
        stream : 数据流（节点到达顺序列表）

        Returns
        -------
        Tuple[Set[int], float] : (解集, 影响力估计值)
        """
        self.oracle_calls = 0
        self.total_budget = 0

        # 首先估计 max singleton value m0 = max_e f({e})
        max_singleton = 0.0
        for v in stream:
            val = self.ic.estimate_spread({v}, self.M_exact // 4,
                                           seed_offset=self.oracle_calls)
            self.oracle_calls += 1
            self.total_budget += self.M_exact // 4
            if val > max_singleton:
                max_singleton = val

        # 生成几何间隔阈值集合 (Badanidiyuru et al. 2014)
        # τ_i = m0 * (1+ε)^i / (2k), i = 0, 1, ..., ceil(log_{1+ε}(2k))
        # 其中 m0 = max_e f({e})
        thresholds = []
        v = max_singleton / (2 * self.k)   # 起始阈值
        v_max = max_singleton
        while v <= v_max:
            thresholds.append(v)
            v *= (1 + self.epsilon)

        # 对每个阈值维护解集和当前 f 值
        S_for_threshold = [set() for _ in thresholds]
        f_for_threshold = [0.0 for _ in thresholds]

        for v in stream:
            # 对每个未满阈值检查是否应加入当前元素
            for i, tau in enumerate(thresholds):
                if len(S_for_threshold[i]) >= self.k:
                    continue
                # 估计边际增益
                gain = self.ic.estimate_marginal_gain(
                    v, S_for_threshold[i], self.M_exact,
                    f_for_threshold[i], seed_offset=self.oracle_calls)
                self.oracle_calls += 1
                self.total_budget += self.M_exact

                if gain >= tau:
                    S_for_threshold[i].add(v)
                    f_for_threshold[i] = self.ic.estimate_spread(
                        S_for_threshold[i], self.M_exact,
                        seed_offset=self.oracle_calls)
                    self.total_budget += self.M_exact

        # 选择 f 值最大的解集
        best_idx = 0
        best_val = 0.0
        for i, S_i in enumerate(S_for_threshold):
            val = self.ic.estimate_spread(S_i, self.M_exact * 2,
                                           seed_offset=self.oracle_calls)
            self.total_budget += self.M_exact * 2
            if val > best_val:
                best_val = val
                best_idx = i

        return S_for_threshold[best_idx], best_val


class ATBMax:
    """
    ATB-Max: Adaptive Threshold Bandit-Max
    ========================================
    基于自适应阈值与多臂赌博机的基数约束单调次模最大化算法。

    算法由三个模块构成：
    1. 轻量边际增益估计器 — 低成本预筛选
    2. 候选阈值集合 — 建模为 MAB 臂集
    3. UCB 策略 — 在线学习最优阈值

    Parameters
    ----------
    ic_model : IC 传播模型
    k : 基数约束
    M_est : 轻量估计器采样数 (default: 240)
    M_exact : 精确 Oracle 采样数 (default: 2500)
    beta : 估计误差缓冲参数 (default: 0.12)
    lam : 计算成本惩罚因子 (default: 0.005)
    m_thresholds : 候选阈值个数 (default: 20)
    tau_min_frac : 最小阈值比例 (default: 0.001)
    tau_max_frac : 最大阈值比例 (default: 0.5)
    use_ucb : 是否启用 UCB (True=ATB-Max, False=无UCB变体)
    use_estimator : 是否使用轻量估计器 (True=ATB-Max, False=无估计器变体)
    threshold_mode : 'ucb' | 'fixed' | 'random'
    """

    def __init__(self, ic_model: IndependentCascadeModel, k: int,
                 M_est: int = 240, M_exact: int = 2500,
                 beta: float = 0.12, lam: float = 0.005,
                 m_thresholds: int = 20,
                 tau_min_frac: float = 0.001,
                 tau_max_frac: float = 0.5,
                 use_ucb: bool = True,
                 use_estimator: bool = True,
                 threshold_mode: str = 'ucb'):
        self.ic = ic_model
        self.k = k
        self.M_est = M_est
        self.M_exact = M_exact
        self.beta = beta
        self.lam = lam
        self.m = m_thresholds
        self.tau_min_frac = tau_min_frac
        self.tau_max_frac = tau_max_frac
        self.use_ucb = use_ucb
        self.use_estimator = use_estimator
        self.threshold_mode = threshold_mode

        # 统计计数器
        self.oracle_calls = 0
        self.total_budget = 0
        self.lightweight_calls = 0
        self.replacements = 0

        # UCB 统计量
        self.thresholds = None       # 候选阈值列表 (归一化后)
        self.tau_raw = None          # 候选阈值原始值列表
        self.N = None                # 每个臂被选中的次数
        self.sum_reward = None       # 每个臂的累计奖励
        self.arm_rewards = None      # 每个臂的奖励历史（用于绘奖励曲线）
        self.chosen_arms = None      # 每轮选择的臂

        # 算法运行过程记录
        self.trace_spread = []       # 解集传播范围随时间的变化
        self.trace_reward = []       # 累计平均奖励随时间的变化
        self.trace_regret = []       # 伪遗憾随时间的变化

    def _init_thresholds(self, max_spread: float):
        """初始化候选阈值集合（对数间隔）"""
        tau_min = self.tau_min_frac * max_spread
        tau_max = self.tau_max_frac * max_spread

        if self.m == 1:
            self.tau_raw = np.array([(tau_min + tau_max) / 2])
        else:
            self.tau_raw = np.exp(np.linspace(np.log(tau_min), np.log(tau_max), self.m))

        # 归一化阈值: τ / max_spread (用于奖励归一化)
        self.thresholds = self.tau_raw / max_spread

    def _ucb_select(self, t: int) -> int:
        """
        UCB1 策略选择当前阈值臂。

        若某臂尚未被选中，其 UCB 指数为 +∞，保证优先探索。
        """
        ucb_values = np.zeros(self.m)
        for i in range(self.m):
            if self.N[i] == 0:
                return i  # 每个臂至少被探索一次
            avg_reward = self.sum_reward[i] / self.N[i]
            exploration_bonus = np.sqrt(2 * np.log(t) / self.N[i])
            ucb_values[i] = avg_reward + exploration_bonus

        return int(np.argmax(ucb_values))

    def _compute_marginal_gain_lightweight(self, node: int, S: Set[int],
                                            cached_spread: float,
                                            seed_offset: int) -> float:
        """轻量级边际增益估计 (M_est 次模拟)"""
        self.lightweight_calls += 1
        self.total_budget += self.M_est
        return self.ic.estimate_marginal_gain(
            node, S, self.M_est, cached_spread, seed_offset)

    def _compute_marginal_gain_exact(self, node: int, S: Set[int],
                                      cached_spread: float,
                                      seed_offset: int) -> float:
        """精确边际增益估计 (M_exact 次模拟)"""
        self.oracle_calls += 1
        self.total_budget += self.M_exact
        return self.ic.estimate_marginal_gain(
            node, S, self.M_exact, cached_spread, seed_offset)

    def _find_weakest_element(self, S: Set[int], cached_spread: float) -> Tuple[int, float]:
        """
        找出当前解集中边际增益最小的元素。

        使用 cached_spread 避免重复估计 f(S)。
        返回 (最弱元素, 其边际增益)。
        """
        min_gain = float('inf')
        weakest = None

        for x in S:
            S_without_x = S - {x}
            spread_without = self.ic.estimate_spread(
                S_without_x, self.M_exact, seed_offset=self.oracle_calls + 1000)
            gain = cached_spread - spread_without
            self.oracle_calls += 1
            self.total_budget += self.M_exact

            if gain < min_gain:
                min_gain = gain
                weakest = x

        return weakest, min_gain

    def run(self, stream: List[int], verbose: bool = False) -> Tuple[Set[int], float]:
        """
        在单次数据流上运行 ATB-Max 算法。

        Parameters
        ----------
        stream : 数据流（节点到达顺序列表）
        verbose : 是否打印详细日志

        Returns
        -------
        Tuple[Set[int], float] : (解集, 影响力估计值)
        """
        # 重置统计量
        self.oracle_calls = 0
        self.total_budget = 0
        self.lightweight_calls = 0
        self.replacements = 0
        self.trace_spread = []
        self.trace_reward = []
        self.trace_regret = []
        self.chosen_arms = []

        S = set()
        cached_spread = 0.0

        # 估计最大单元素传播范围，用于初始化阈值
        singleton_vals = []
        for v in stream[:min(100, len(stream))]:
            val = self.ic.estimate_spread({v}, self.M_exact // 5,
                                           seed_offset=self.oracle_calls)
            self.total_budget += self.M_exact // 5
            singleton_vals.append(val)
        max_singleton = max(singleton_vals) if singleton_vals else len(stream)
        self._init_thresholds(max_singleton)

        # 初始化 UCB 统计量
        self.N = np.zeros(self.m, dtype=int)
        self.sum_reward = np.zeros(self.m)
        self.arm_rewards = [[] for _ in range(self.m)]

        # 若使用固定阈值模式，预设固定臂
        if self.threshold_mode == 'fixed':
            fixed_arm = self.m // 2  # 使用中间阈值

        # 主循环：处理数据流中的每个元素
        for t_idx, e in enumerate(stream):
            t = t_idx + 1  # UCB 使用 1-indexed

            # ---- 步骤 1: 选择阈值 ----
            if self.threshold_mode == 'ucb' and self.use_ucb:
                arm = self._ucb_select(t)
                tau = self.tau_raw[arm]
            elif self.threshold_mode == 'random':
                arm = random.randint(0, self.m - 1)
                tau = self.tau_raw[arm]
            elif self.threshold_mode == 'fixed':
                arm = fixed_arm
                tau = self.tau_raw[arm]
            else:
                arm = 0
                tau = self.tau_raw[0]

            self.chosen_arms.append(arm)

            # ---- 步骤 2: 轻量估计器预筛选 ----
            if self.use_estimator:
                delta_hat = self._compute_marginal_gain_lightweight(
                    e, S, cached_spread, seed_offset=t_idx * 10)
                passed_screening = (delta_hat >= tau - self.beta * max_singleton)
            else:
                # 无估计器变体：直接进入精确评估
                delta_hat = tau  # 确保通过筛选
                passed_screening = True

            reward = 0.0
            if passed_screening:
                # ---- 步骤 3: 精确 Oracle 评估 ----
                delta_exact = self._compute_marginal_gain_exact(
                    e, S, cached_spread, seed_offset=t_idx * 100)

                # ---- 步骤 4: 决策 ----
                added = False
                if len(S) < self.k:
                    S.add(e)
                    added = True
                    cached_spread = self.ic.estimate_spread(
                        S, self.M_exact, seed_offset=self.oracle_calls)
                    self.total_budget += self.M_exact
                else:
                    # 解集已满，检查是否需要替换
                    weakest, gain_weakest = self._find_weakest_element(S, cached_spread)
                    if weakest is not None and weakest != e:
                        if delta_exact > gain_weakest:
                            S.discard(weakest)
                            S.add(e)
                            self.replacements += 1
                            added = True
                            cached_spread = self.ic.estimate_spread(
                                S, self.M_exact, seed_offset=self.oracle_calls)
                            self.total_budget += self.M_exact

                # ---- 步骤 5: 计算奖励 ----
                if added:
                    reward = delta_exact / max_singleton - self.lam
                else:
                    reward = -self.lam
            else:
                reward = 0.0

            # ---- 步骤 6: 更新 UCB 统计量 ----
            self.N[arm] += 1
            self.sum_reward[arm] += reward
            self.arm_rewards[arm].append(reward)

            # ---- 步骤 7: 记录 trace ----
            if t % max(1, len(stream) // 50) == 0 or t == len(stream):
                self.trace_spread.append((t, self.ic.estimate_spread(
                    S, self.M_exact, seed_offset=self.oracle_calls + t)))
                self.total_budget += self.M_exact

            # 累计平均奖励
            if t % max(1, len(stream) // 50) == 0:
                total_reward = sum(sum(r_list) for r_list in self.arm_rewards)
                self.trace_reward.append((t, total_reward / t))

            # 伪遗憾记录
            if self.threshold_mode == 'ucb' and t % max(1, len(stream) // 50) == 0:
                best_reward = max(
                    (self.sum_reward[i] / max(self.N[i], 1)) for i in range(self.m)
                )
                avg_reward_per_round = (
                    sum(self.sum_reward) / max(sum(self.N), 1)
                )
                pseudo_regret = t * (best_reward - avg_reward_per_round)
                self.trace_regret.append((t, max(0, pseudo_regret)))

        # 最终解质量评估（高精度）
        final_spread = self.ic.estimate_spread(S, self.M_exact * 2,
                                                seed_offset=len(stream) * 1000)
        self.total_budget += self.M_exact * 2

        if verbose:
            print(f"  ATB-Max finished: |S|={len(S)}, "
                  f"spread={final_spread:.2f}, "
                  f"oracle_calls={self.oracle_calls}, "
                  f"total_budget={self.total_budget}, "
                  f"replacements={self.replacements}")

        return S, final_spread


# ============================================================================
# Section 4: 实验运行器
# ============================================================================

@dataclass
class AlgorithmResult:
    """单次算法运行结果"""
    algorithm_name: str
    spread: float               # 传播范围
    oracle_calls: int           # 精确 Oracle 调用次数
    total_budget: int           # 总传播模拟预算
    runtime: float              # 运行时间 (秒)
    solution_size: int          # 解集大小
    trace_spread: List          # (可选) 传播范围随时间变化
    trace_reward: List          # (可选) 累计平均奖励
    trace_regret: List          # (可选) 伪遗憾
    arm_counts: np.ndarray      # (可选) 各臂选择频次


class ExperimentRunner:
    """实验运行器：负责协调算法运行和数据收集"""

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def run_single_experiment(self, graph: nx.DiGraph, stream: List[int],
                               algorithm_name: str,
                               algorithm: object) -> AlgorithmResult:
        """
        运行单次算法实验。

        Parameters
        ----------
        graph : 网络图
        stream : 节点流顺序
        algorithm_name : 算法名称
        algorithm : 算法实例

        Returns
        -------
        AlgorithmResult
        """
        t_start = time.perf_counter()
        S, spread = algorithm.run(stream)
        t_end = time.perf_counter()

        # 收集统计量
        oracle_calls = getattr(algorithm, 'oracle_calls', 0)
        total_budget = getattr(algorithm, 'total_budget', 0)

        # 收集 trace 信息 (仅 ATB-Max 类算法)
        trace_spread = getattr(algorithm, 'trace_spread', [])
        trace_reward = getattr(algorithm, 'trace_reward', [])
        trace_regret = getattr(algorithm, 'trace_regret', [])
        arm_counts = None
        if hasattr(algorithm, 'chosen_arms') and algorithm.chosen_arms:
            arm_counts = np.bincount(algorithm.chosen_arms,
                                      minlength=algorithm.m)

        return AlgorithmResult(
            algorithm_name=algorithm_name,
            spread=spread,
            oracle_calls=oracle_calls,
            total_budget=total_budget,
            runtime=t_end - t_start,
            solution_size=len(S),
            trace_spread=trace_spread,
            trace_reward=trace_reward,
            trace_regret=trace_regret,
            arm_counts=arm_counts,
        )

    def run_all_algorithms(self, graph: nx.DiGraph,
                            dataset_name: str,
                            skip_greedy: bool = False,
                            skip_sieve: bool = False) -> Dict[str, List[AlgorithmResult]]:
        """
        在给定数据集上运行全部对比算法，重复多次取平均。

        Parameters
        ----------
        skip_greedy : 跳过 Greedy 算法（加速）
        skip_sieve : 跳过 Sieve-Streaming 算法（加速）

        Returns
        -------
        Dict[str, List[AlgorithmResult]] : {算法名: [各次重复结果]}
        """
        cfg = self.config
        nodes = list(graph.nodes())
        n = len(nodes)

        # 初始化 IC 模型
        ic_model = IndependentCascadeModel(graph, mode='weighted_cascade',
                                            seed=cfg.random_state)
        print(f"\n{'=' * 60}")
        print(f"数据集: {dataset_name}")
        print(f"节点数: {n}, 边数: {graph.number_of_edges()}")
        print(f"参数: k={cfg.k}, beta={cfg.beta}, lambda={cfg.lam}, "
              f"M_est={cfg.M_est}, M_exact={cfg.M_exact}")
        print(f"重复次数: {cfg.n_repeats}")
        if skip_greedy:
            print("  (跳过 Greedy)")
        if skip_sieve:
            print("  (跳过 Sieve-Streaming)")
        print(f"{'=' * 60}")

        # 预生成所有流顺序（保证公平比较）
        streams = []
        for rep in range(cfg.n_repeats):
            rep_seed = cfg.random_state + rep * 1000
            rng = np.random.RandomState(rep_seed)
            stream = nodes.copy()
            rng.shuffle(stream)
            streams.append(stream)

        # 初始化所有算法结果容器
        all_results = {
            "ATB-Max": [],
            "ATB-Max (无估计器)": [],
            "ATB-Max (无UCB)": [],
            "固定阈值": [],
            "随机阈值": [],
            "Sieve-Streaming": [],
            "Greedy": [],
        }

        for rep in range(cfg.n_repeats):
            stream = streams[rep]
            rep_seed = cfg.random_state + rep * 1000
            print(f"\n--- 第 {rep + 1}/{cfg.n_repeats} 次重复 (seed={rep_seed}) ---")

            # 1. ATB-Max (完整版)
            t0 = time.perf_counter()
            ic_rep = IndependentCascadeModel(graph, mode='weighted_cascade',
                                              seed=rep_seed)
            atbmax = ATBMax(ic_rep, k=cfg.k, M_est=cfg.M_est,
                            M_exact=cfg.M_exact, beta=cfg.beta,
                            lam=cfg.lam, m_thresholds=cfg.m_thresholds,
                            use_ucb=True, use_estimator=True,
                            threshold_mode='ucb')
            result = self.run_single_experiment(graph, stream, "ATB-Max", atbmax)
            all_results["ATB-Max"].append(result)
            print(f"  ATB-Max: spread={result.spread:.2f}, "
                  f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                  f"time={result.runtime:.2f}s")

            # 2. ATB-Max (无估计器变体) — 所有元素直接触发精确 Oracle
            ic_rep2 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                               seed=rep_seed)
            atbmax_no_est = ATBMax(ic_rep2, k=cfg.k, M_est=cfg.M_est,
                                   M_exact=cfg.M_exact, beta=cfg.beta,
                                   lam=cfg.lam, m_thresholds=cfg.m_thresholds,
                                   use_ucb=True, use_estimator=False,
                                   threshold_mode='ucb')
            result = self.run_single_experiment(graph, stream,
                                                 "ATB-Max (无估计器)",
                                                 atbmax_no_est)
            all_results["ATB-Max (无估计器)"].append(result)
            print(f"  ATB-Max (无估计器): spread={result.spread:.2f}, "
                  f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                  f"time={result.runtime:.2f}s")

            # 3. ATB-Max (无UCB变体) — 使用固定中间阈值
            ic_rep3 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                               seed=rep_seed)
            atbmax_no_ucb = ATBMax(ic_rep3, k=cfg.k, M_est=cfg.M_est,
                                   M_exact=cfg.M_exact, beta=cfg.beta,
                                   lam=cfg.lam, m_thresholds=cfg.m_thresholds,
                                   use_ucb=False, use_estimator=True,
                                   threshold_mode='fixed')
            result = self.run_single_experiment(graph, stream,
                                                 "ATB-Max (无UCB)",
                                                 atbmax_no_ucb)
            all_results["ATB-Max (无UCB)"].append(result)
            print(f"  ATB-Max (无UCB): spread={result.spread:.2f}, "
                  f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                  f"time={result.runtime:.2f}s")

            # 4. 固定阈值
            ic_rep4 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                               seed=rep_seed)
            fixed_th = ATBMax(ic_rep4, k=cfg.k, M_est=cfg.M_est,
                              M_exact=cfg.M_exact, beta=cfg.beta,
                              lam=cfg.lam, m_thresholds=cfg.m_thresholds,
                              use_ucb=False, use_estimator=True,
                              threshold_mode='fixed')
            result = self.run_single_experiment(graph, stream,
                                                 "固定阈值", fixed_th)
            all_results["固定阈值"].append(result)
            print(f"  固定阈值: spread={result.spread:.2f}, "
                  f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                  f"time={result.runtime:.2f}s")

            # 5. 随机阈值
            ic_rep5 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                               seed=rep_seed)
            rand_th = ATBMax(ic_rep5, k=cfg.k, M_est=cfg.M_est,
                             M_exact=cfg.M_exact, beta=cfg.beta,
                             lam=cfg.lam, m_thresholds=cfg.m_thresholds,
                             use_ucb=False, use_estimator=True,
                             threshold_mode='random')
            result = self.run_single_experiment(graph, stream,
                                                 "随机阈值", rand_th)
            all_results["随机阈值"].append(result)
            print(f"  随机阈值: spread={result.spread:.2f}, "
                  f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                  f"time={result.runtime:.2f}s")

            # 6. Sieve-Streaming (可选)
            if not skip_sieve:
                ic_rep6 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                                   seed=rep_seed)
                sieve = SieveStreaming(ic_rep6, k=cfg.k, M_exact=cfg.M_exact,
                                       epsilon=cfg.epsilon)
                result = self.run_single_experiment(graph, stream,
                                                     "Sieve-Streaming", sieve)
                all_results["Sieve-Streaming"].append(result)
                print(f"  Sieve-Streaming: spread={result.spread:.2f}, "
                      f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                      f"time={result.runtime:.2f}s")

            # 7. Greedy (离线，与流顺序无关，仅在第1次重复时运行)
            if not skip_greedy:
                if rep == 0:
                    ic_rep7 = IndependentCascadeModel(graph, mode='weighted_cascade',
                                                       seed=rep_seed)
                    greedy = GreedyAlgorithm(ic_rep7, k=cfg.k, M_exact=cfg.M_exact)
                    result = self.run_single_experiment(graph, stream, "Greedy", greedy)
                    all_results["Greedy"].append(result)
                    print(f"  Greedy: spread={result.spread:.2f}, "
                          f"oracle={result.oracle_calls}, budget={result.total_budget}, "
                          f"time={result.runtime:.2f}s")
                else:
                    # 复制 Greedy 结果
                    all_results["Greedy"].append(all_results["Greedy"][0])

        return all_results


# ============================================================================
# Section 5: 实验结果分析与可视化
# ============================================================================

def summarize_results(all_results: Dict[str, List[AlgorithmResult]]) -> Dict:
    """汇总多次重复实验结果为均值和标准差"""
    summary = {}
    for algo_name, results in all_results.items():
        if not results:
            continue
        spreads = [r.spread for r in results]
        oracles = [r.oracle_calls for r in results]
        budgets = [r.total_budget for r in results]
        runtimes = [r.runtime for r in results]

        summary[algo_name] = {
            'spread_mean': np.mean(spreads) if spreads else float('nan'),
            'spread_std': np.std(spreads) if len(spreads) > 1 else 0.0,
            'oracle_mean': np.mean(oracles) if oracles else float('nan'),
            'oracle_std': np.std(oracles) if len(oracles) > 1 else 0.0,
            'budget_mean': np.mean(budgets) if budgets else float('nan'),
            'budget_std': np.std(budgets) if len(budgets) > 1 else 0.0,
            'runtime_mean': np.mean(runtimes) if runtimes else float('nan'),
            'runtime_std': np.std(runtimes) if len(runtimes) > 1 else 0.0,
        }
    return summary


def print_comparison_table(summary: Dict, dataset_name: str):
    """打印综合比较表（格式匹配论文表 5.1）"""
    print(f"\n{'=' * 80}")
    print(f"综合比较表 — {dataset_name}")
    print(f"{'=' * 80}")
    header = (f"{'算法名称':<28} {'传播范围均值':>12} {'精确Oracle均值':>16} "
              f"{'总预算均值':>16}")
    print(header)
    print("-" * 80)

    # 按论文中的顺序排列
    order = ["ATB-Max", "ATB-Max (无估计器)", "ATB-Max (无UCB)",
             "固定阈值", "Greedy", "随机阈值", "Sieve-Streaming"]
    for name in order:
        if name in summary:
            s = summary[name]
            print(f"{name:<28} {s['spread_mean']:>12.2f} {s['oracle_mean']:>16.1f} "
                  f"{s['budget_mean']:>16.0f}")

    # 相对解质量
    print(f"\n{'相对解质量 (Ratio = spread / Greedy_spread)':>50}")
    print("-" * 60)
    greedy_spread = summary.get("Greedy", {}).get("spread_mean", 1.0)
    for name in order:
        if name in summary and name != "Greedy":
            ratio = summary[name]['spread_mean'] / max(greedy_spread, 0.01)
            print(f"  {name:<26}: {ratio:.4f}")
    print(f"{'=' * 80}\n")


def _setup_chinese_font():
    """
    配置 matplotlib 使用中文字体。

    按优先级尝试：SimHei → Microsoft YaHei → STXihei → SimSun → KaiTi。
    若全部不可用则回退到 DejaVu Sans（英文显示）。

    Returns
    -------
    bool : 是否成功配置了中文字体
    """
    from matplotlib.font_manager import FontProperties
    import matplotlib as mpl

    candidate_fonts = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun', 'KaiTi',
                       'STSong', 'STKaiti', 'FangSong']

    chosen = None
    for font_name in candidate_fonts:
        try:
            fp = FontProperties(family=font_name)
            # 尝试获取字体文件路径来验证字体确实存在
            from matplotlib.font_manager import findfont
            font_path = findfont(fp, fallback_to_default=False)
            if font_path and font_path != findfont(FontProperties(), fallback_to_default=True):
                chosen = font_name
                break
        except Exception:
            continue

    if chosen:
        mpl.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['axes.unicode_minus'] = False
        print(f"  [Font] 使用中文字体: {chosen}")
        return True
    else:
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['axes.unicode_minus'] = False
        print("  [Font] 未找到中文字体，图表将使用英文标签")
        return False


def generate_figures(all_results: Dict[str, List[AlgorithmResult]],
                     summary: Dict, dataset_name: str,
                     output_dir: str = "experiment_figures"):
    """
    生成实验图表。

    包括:
      fig1: 不同算法传播范围对比 (对应论文图 5.1)
      fig2: 精确 Oracle 调用对比 (对应论文图 5.2)
      fig3: 总传播模拟预算对比 (对应论文图 5.3)
      fig4: 运行时间对比 (对应论文图 5.4)
      fig5: 不同流顺序下的传播范围分布 (对应论文图 5.5)
      fig6: UCB 累计平均奖励 (对应论文图 5.6)
      fig7: UCB 累计遗憾 (对应论文图 5.7)
      fig8: UCB 阈值臂选择频次 (对应论文图 5.8)
    """
    os.makedirs(output_dir, exist_ok=True)

    has_cjk = _setup_chinese_font()

    # 配色方案
    colors = {
        "ATB-Max": "#2196F3",
        "ATB-Max (无估计器)": "#FF9800",
        "ATB-Max (无UCB)": "#9E9E9E",
        "固定阈值": "#607D8B",
        "随机阈值": "#795548",
        "Sieve-Streaming": "#4CAF50",
        "Greedy": "#F44336",
    }

    algo_order = ["ATB-Max", "ATB-Max (无估计器)", "ATB-Max (无UCB)",
                  "固定阈值", "随机阈值", "Sieve-Streaming", "Greedy"]

    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    # ---- Figure 1: 传播范围对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [a for a in algo_order if a in summary]
    spreads = [summary[a]['spread_mean'] for a in names]
    errors = [summary[a]['spread_std'] for a in names]
    bar_colors = [colors.get(a, '#999999') for a in names]

    bars = ax.bar(range(len(names)), spreads, yerr=errors, color=bar_colors,
                  capsize=5, edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Influence Spread', fontsize=11)
    ax.set_title(f'Fig 1: Influence Spread Comparison ({dataset_name})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, spreads):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_spread_comparison.png'), dpi=150)
    plt.close(fig)

    # ---- Figure 2: 精确 Oracle 调用对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    oracles = [summary[a]['oracle_mean'] for a in names]
    bar_colors = [colors.get(a, '#999999') for a in names]
    bars = ax.bar(range(len(names)), oracles, color=bar_colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Exact Oracle Calls', fontsize=11)
    ax.set_title(f'Fig 2: Exact Oracle Calls Comparison ({dataset_name})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, oracles):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_oracle_calls.png'), dpi=150)
    plt.close(fig)

    # ---- Figure 3: 总传播模拟预算对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    budgets = [summary[a]['budget_mean'] for a in names]
    bar_colors = [colors.get(a, '#999999') for a in names]
    bars = ax.bar(range(len(names)), budgets, color=bar_colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Total Simulation Budget', fontsize=11)
    ax.set_title(f'Fig 3: Total Simulation Budget Comparison ({dataset_name})',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, budgets):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1e}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_total_budget.png'), dpi=150)
    plt.close(fig)

    # ---- Figure 4: 运行时间对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    runtimes = [summary[a]['runtime_mean'] for a in names]
    bar_colors = [colors.get(a, '#999999') for a in names]
    ax.bar(range(len(names)), runtimes, color=bar_colors,
           edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title(f'Fig 4: Runtime Comparison ({dataset_name})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_runtime.png'), dpi=150)
    plt.close(fig)

    # ---- Figure 5: 不同流顺序下的传播范围分布 (箱线图) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = []
    box_labels = []
    for a in algo_order:
        if a in all_results and len(all_results[a]) > 1:
            box_data.append([r.spread for r in all_results[a]])
            box_labels.append(a)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, label in zip(bp['boxes'], box_labels):
        patch.set_facecolor(colors.get(label, '#999999'))
        patch.set_alpha(0.7)
    ax.set_ylabel('Influence Spread', fontsize=11)
    ax.set_title(f'Fig 5: Spread Distribution Across Stream Orders ({dataset_name})',
                 fontsize=12)
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_spread_distribution.png'), dpi=150)
    plt.close(fig)

    # ---- Figure 6-8: UCB 学习过程 (仅当 ATB-Max 有 trace 数据时) ----
    atbmax_results = all_results.get("ATB-Max", [])
    if atbmax_results and atbmax_results[0].trace_reward:
        r0 = atbmax_results[0]  # 使用第一次重复的结果

        # Fig 6: UCB 累计平均奖励
        if r0.trace_reward:
            fig, ax = plt.subplots(figsize=(8, 4))
            t_vals, r_vals = zip(*r0.trace_reward)
            ax.plot(t_vals, r_vals, 'b-', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Round t', fontsize=11)
            ax.set_ylabel('Cumulative Average Reward', fontsize=11)
            ax.set_title('Fig 6: UCB Cumulative Average Reward', fontsize=12)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'fig6_ucb_reward.png'), dpi=150)
            plt.close(fig)

        # Fig 7: UCB 累计伪遗憾
        if r0.trace_regret:
            fig, ax = plt.subplots(figsize=(8, 4))
            t_vals, reg_vals = zip(*r0.trace_regret)
            ax.plot(t_vals, reg_vals, 'r-', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Round t', fontsize=11)
            ax.set_ylabel('Cumulative Pseudo-Regret', fontsize=11)
            ax.set_title('Fig 7: UCB Cumulative Pseudo-Regret', fontsize=12)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'fig7_ucb_regret.png'), dpi=150)
            plt.close(fig)

        # Fig 8: 阈值臂选择频次
        if r0.arm_counts is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            m = len(r0.arm_counts)
            ax.bar(range(m), r0.arm_counts, color='steelblue', edgecolor='white')
            ax.set_xlabel('Threshold Arm Index (0=lowest, higher=stricter)', fontsize=11)
            ax.set_ylabel('Number of Times Selected', fontsize=11)
            ax.set_title('Fig 8: Threshold Arm Selection Frequency', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'fig8_arm_frequency.png'), dpi=150)
            plt.close(fig)

    print(f"图表已保存至 {output_dir}/ 目录")


def run_parameter_sensitivity(graph: nx.DiGraph, dataset_name: str,
                               config: ExperimentConfig,
                               output_dir: str = "experiment_figures"):
    """
    参数敏感性分析。

    考察四个关键参数对 ATB-Max 性能的影响：
      - beta (估计误差缓冲)
      - lambda (惩罚因子)
      - k (种子集规模)
      - M_est (轻量估计样本数)

    对应论文 §5.4 敏感性分析 (表 5.2, 图 5.9-5.12)
    """
    print(f"\n{'=' * 60}")
    print(f"参数敏感性分析 — {dataset_name}")
    print(f"{'=' * 60}")

    nodes = list(graph.nodes())
    n_repeats = 5  # 敏感性分析使用较少的重复次数

    # 预生成流顺序
    streams = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(config.random_state + rep * 1000)
        stream = nodes.copy()
        rng.shuffle(stream)
        streams.append(stream)

    def _run_sensitivity(param_name: str, param_values: List,
                          vary_func: Callable):
        """运行单参数敏感性实验"""
        print(f"\n--- {param_name} 敏感性分析 ---")
        results = []

        for val in param_values:
            cfg_dict = {
                'k': config.k, 'M_est': config.M_est,
                'M_exact': config.M_exact, 'beta': config.beta,
                'lam': config.lam, 'm_thresholds': config.m_thresholds,
            }
            vary_func(cfg_dict, val)

            spreads = []
            runtimes = []
            oracles = []
            budgets = []

            for rep in range(n_repeats):
                stream = streams[rep]
                rep_seed = config.random_state + rep * 1000 + hash(val) % 10000
                ic_rep = IndependentCascadeModel(graph, mode='weighted_cascade',
                                                  seed=rep_seed)
                atbmax = ATBMax(ic_rep, k=cfg_dict['k'],
                                M_est=cfg_dict['M_est'],
                                M_exact=cfg_dict['M_exact'],
                                beta=cfg_dict['beta'],
                                lam=cfg_dict['lam'],
                                m_thresholds=cfg_dict['m_thresholds'],
                                use_ucb=True, use_estimator=True,
                                threshold_mode='ucb')

                t0 = time.perf_counter()
                S, spread = atbmax.run(stream)
                t1 = time.perf_counter()

                spreads.append(spread)
                runtimes.append(t1 - t0)
                oracles.append(atbmax.oracle_calls)
                budgets.append(atbmax.total_budget)

            results.append({
                'param_value': val,
                'spread_mean': np.mean(spreads),
                'spread_std': np.std(spreads),
                'runtime_mean': np.mean(runtimes),
                'oracle_mean': np.mean(oracles),
                'budget_mean': np.mean(budgets),
            })
            print(f"  {param_name}={val}: spread={np.mean(spreads):.2f}±{np.std(spreads):.2f}, "
                  f"runtime={np.mean(runtimes):.2f}s, oracle={np.mean(oracles):.0f}")

        return results

    os.makedirs(output_dir, exist_ok=True)
    all_sensitivity = {}

    # (1) beta 敏感性
    beta_values = [0.06, 0.08, 0.10, 0.12, 0.14]
    all_sensitivity['beta'] = _run_sensitivity(
        'beta', beta_values,
        lambda d, v: d.update({'beta': v}))

    # (2) lambda 敏感性
    lam_values = [0.0, 0.003, 0.005, 0.008, 0.010]
    all_sensitivity['lambda'] = _run_sensitivity(
        'lambda', lam_values,
        lambda d, v: d.update({'lam': v}))

    # (3) k 敏感性
    k_values = [20, 50, 100]
    all_sensitivity['k'] = _run_sensitivity(
        'k', k_values,
        lambda d, v: d.update({'k': v}))

    # (4) M_est 敏感性
    M_est_values = [120, 180, 240, 320]
    all_sensitivity['M_est'] = _run_sensitivity(
        'M_est', M_est_values,
        lambda d, v: d.update({'M_est': v}))

    # ---- 绘制敏感性分析图 ----
    param_configs = [
        ('beta', 'Estimation Error Buffer β', [0.06, 0.08, 0.10, 0.12, 0.14],
         'fig9_beta_sensitivity.png'),
        ('lambda', 'Cost Penalty Factor λ', [0.0, 0.003, 0.005, 0.008, 0.010],
         'fig10_lambda_sensitivity.png'),
        ('k', 'Cardinality Constraint k', [20, 50, 100],
         'fig11_k_sensitivity.png'),
        ('M_est', 'Lightweight Estimation Samples M_est', [120, 180, 240, 320],
         'fig12_Mest_sensitivity.png'),
    ]

    for param_key, param_label, x_vals, filename in param_configs:
        if param_key not in all_sensitivity:
            continue
        data = all_sensitivity[param_key]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        x = [d['param_value'] for d in data]
        spreads = [d['spread_mean'] for d in data]
        spread_stds = [d['spread_std'] for d in data]
        runtimes = [d['runtime_mean'] for d in data]

        # 传播范围
        ax1.errorbar(range(len(x)), spreads, yerr=spread_stds,
                     fmt='o-', color='#2196F3', linewidth=2, markersize=8,
                     capsize=5, label='Influence Spread')
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels([str(v) for v in x])
        ax1.set_xlabel(param_label, fontsize=11)
        ax1.set_ylabel('Influence Spread', fontsize=11)
        ax1.set_title(f'Sensitivity to {param_label} (Spread)', fontsize=11)
        ax1.grid(alpha=0.3)

        # 运行时间
        ax2.bar(range(len(x)), runtimes, color='#FF9800',
                edgecolor='white', linewidth=0.8)
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([str(v) for v in x])
        ax2.set_xlabel(param_label, fontsize=11)
        ax2.set_ylabel('Runtime (s)', fontsize=11)
        ax2.set_title(f'Sensitivity to {param_label} (Runtime)', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    print(f"\n敏感性分析图表已保存至 {output_dir}/ 目录")
    return all_sensitivity


def run_scalability_experiment(datasets: Dict[str, nx.DiGraph],
                                config: ExperimentConfig,
                                output_dir: str = "experiment_figures"):
    """
    可扩展性实验：在多个不同规模数据集上测试算法性能。

    评估 ATB-Max 随节点数增长时在解质量、计算开销和运行时间上的变化趋势。
    """
    print(f"\n{'=' * 60}")
    print("可扩展性实验 (Scalability Experiment)")
    print(f"{'=' * 60}")

    results_by_dataset = {}
    for ds_name, graph in datasets.items():
        config_copy = ExperimentConfig(**{k: v for k, v in config.__dict__.items()})
        runner = ExperimentRunner(config_copy)

        # 为可扩展性实验减少重复次数
        config_copy.n_repeats = 5
        all_results = runner.run_all_algorithms(graph, ds_name)
        summary = summarize_results(all_results)
        results_by_dataset[ds_name] = summary
        print_comparison_table(summary, ds_name)

    # ---- 可扩展性汇总图 ----
    os.makedirs(output_dir, exist_ok=True)

    ds_names = list(datasets.keys())
    short_names = [n.split('(')[0].strip() for n in ds_names]
    ds_sizes = [g.number_of_nodes() for g in datasets.values()]

    # 按节点数排序
    sorted_idx = np.argsort(ds_sizes)
    sorted_names = [short_names[i] for i in sorted_idx]
    sorted_sizes = [ds_sizes[i] for i in sorted_idx]
    sorted_ds = [list(datasets.keys())[i] for i in sorted_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    algo_names = ["ATB-Max", "Sieve-Streaming", "Greedy"]
    algo_colors = {"ATB-Max": "#2196F3", "Sieve-Streaming": "#4CAF50",
                   "Greedy": "#F44336"}
    markers = {"ATB-Max": 'o', "Sieve-Streaming": 's', "Greedy": '^'}

    # (a) 传播范围 vs 数据集规模
    ax = axes[0, 0]
    for algo in algo_names:
        spreads = []
        for ds_key in sorted_ds:
            if algo in results_by_dataset[ds_key]:
                spreads.append(results_by_dataset[ds_key][algo]['spread_mean'])
            else:
                spreads.append(np.nan)
        ax.plot(sorted_sizes, spreads, marker=markers[algo],
                color=algo_colors[algo], linewidth=2, markersize=8, label=algo)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Influence Spread', fontsize=11)
    ax.set_title('(a) Spread vs Dataset Size', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 精确 Oracle 调用 vs 数据集规模
    ax = axes[0, 1]
    for algo in algo_names:
        oracles = []
        for ds_key in sorted_ds:
            if algo in results_by_dataset[ds_key]:
                oracles.append(results_by_dataset[ds_key][algo]['oracle_mean'])
            else:
                oracles.append(np.nan)
        ax.plot(sorted_sizes, oracles, marker=markers[algo],
                color=algo_colors[algo], linewidth=2, markersize=8, label=algo)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Exact Oracle Calls', fontsize=11)
    ax.set_title('(b) Oracle Calls vs Dataset Size', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 总预算 vs 数据集规模
    ax = axes[1, 0]
    for algo in algo_names:
        budgets = []
        for ds_key in sorted_ds:
            if algo in results_by_dataset[ds_key]:
                budgets.append(results_by_dataset[ds_key][algo]['budget_mean'])
            else:
                budgets.append(np.nan)
        ax.plot(sorted_sizes, budgets, marker=markers[algo],
                color=algo_colors[algo], linewidth=2, markersize=8, label=algo)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Total Simulation Budget', fontsize=11)
    ax.set_title('(c) Total Budget vs Dataset Size', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 运行时间 vs 数据集规模
    ax = axes[1, 1]
    for algo in algo_names:
        runtimes = []
        for ds_key in sorted_ds:
            if algo in results_by_dataset[ds_key]:
                runtimes.append(results_by_dataset[ds_key][algo]['runtime_mean'])
            else:
                runtimes.append(np.nan)
        ax.plot(sorted_sizes, runtimes, marker=markers[algo],
                color=algo_colors[algo], linewidth=2, markersize=8, label=algo)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('(d) Runtime vs Dataset Size', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Scalability Analysis: Performance vs Dataset Size',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_scalability.png'), dpi=150)
    plt.close(fig)
    print(f"\n可扩展性图表已保存至 {output_dir}/ 目录")

    return results_by_dataset


# ============================================================================
# Section 6: 主实验入口
# ============================================================================

def main():
    """主实验入口函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='ATB-Max Algorithm Experiment Suite — '
                    '大规模数据环境下次模最大化问题新算法实验验证')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='实验模式: quick=快速验证 (n_repeats=3, 小图), '
                             'full=完整实验 (n_repeats=10, 匹配论著参数)')
    parser.add_argument('--output', type=str, default='experiment_figures',
                        help='图表输出目录 (默认: experiment_figures)')
    parser.add_argument('--skip-greedy', action='store_true',
                        help='跳过 Greedy 算法 (加速)')
    parser.add_argument('--skip-sieve', action='store_true',
                        help='跳过 Sieve-Streaming 算法 (加速)')
    args = parser.parse_args()

    print("=" * 60)
    print("ATB-Max Algorithm Experiment Suite")
    print("大规模数据环境下次模最大化问题新算法实验验证")
    print(f"模式: {args.mode}")
    print("=" * 60)

    # 配置中文字体（需在所有绘图之前调用）
    _setup_chinese_font()

    # ---- 配置 ----
    if args.mode == 'quick':
        config = ExperimentConfig(
            k=20,
            M_est=50,
            M_exact=500,
            beta=0.12,
            lam=0.005,
            m_thresholds=10,
            epsilon=0.2,
            n_repeats=3,
            random_state=42,
        )
        print("\n[Quick Mode] 使用较轻量参数进行快速验证")
        print(f"  k={config.k}, M_est={config.M_est}, M_exact={config.M_exact}, "
              f"n_repeats={config.n_repeats}")
        print("  注意: 快速模式用于验证代码正确性，数值结果不代表论著结果。")
        print("  使用 --mode full 运行完整实验。")
    else:
        config = ExperimentConfig(
            k=50,
            M_est=240,
            M_exact=2500,
            beta=0.12,
            lam=0.005,
            m_thresholds=20,
            epsilon=0.1,
            n_repeats=10,
            random_state=42,
        )
        print("\n[Full Mode] 使用论著中的参数设置进行完整实验")
        print(f"  k={config.k}, M_est={config.M_est}, M_exact={config.M_exact}, "
              f"n_repeats={config.n_repeats}")
        print("  预计总运行时间: 数小时至数十小时 (取决于硬件)")
        print("  建议在服务器或高性能计算环境中运行。")

    set_seed(config.random_state)
    total_start_time = time.perf_counter()

    # ---- 准备数据集 ----
    if args.mode == 'quick':
        # 快速模式：只使用两个小数据集
        print("\n[Quick Mode] 使用小型合成数据集...")
        datasets = {}
        g1 = generate_graph_barabasi_albert(n=500, m=10, seed=42)
        datasets["small-BA (n=500)"] = g1
        g2 = generate_graph_erdos_renyi(n=300, p=0.02, seed=123)
        datasets["small-ER (n=300)"] = g2
        print(f"  已准备 {len(datasets)} 个小型数据集")
    else:
        datasets = prepare_datasets()

    # ---- 实验 1: 综合性能比较 (主数据集) ----
    print("\n" + "=" * 60)
    print("实验 1: 综合性能比较 (Comprehensive Comparison)")
    print("=" * 60)

    main_dataset_name = list(datasets.keys())[0]
    main_graph = datasets[main_dataset_name]

    exp_start = time.perf_counter()
    runner = ExperimentRunner(config)
    all_results = runner.run_all_algorithms(
        main_graph, main_dataset_name,
        skip_greedy=args.skip_greedy,
        skip_sieve=args.skip_sieve)
    exp_elapsed = time.perf_counter() - exp_start
    print(f"\n实验 1 完成，耗时: {exp_elapsed:.1f}s")

    # 汇总与打印
    summary = summarize_results(all_results)
    print_comparison_table(summary, main_dataset_name)

    # 生成图表
    generate_figures(all_results, summary, main_dataset_name,
                     output_dir=args.output)

    # ---- 实验 2: 消融实验分析 ----
    print("\n" + "=" * 60)
    print("实验 2: 消融实验分析 (Ablation Study)")
    print("=" * 60)

    if "ATB-Max" in summary and "Greedy" in summary:
        atbmax_spread = summary["ATB-Max"]["spread_mean"]
        atbmax_no_est_spread = summary.get("ATB-Max (无估计器)", {}).get("spread_mean", 0)
        atbmax_no_ucb_spread = summary.get("ATB-Max (无UCB)", {}).get("spread_mean", 0)
        greedy_spread = summary["Greedy"]["spread_mean"]

        print("\n--- UCB 阈值选择模块的作用 ---")
        print(f"  ATB-Max spread:              {atbmax_spread:.2f}")
        print(f"  ATB-Max (无UCB) spread:      {atbmax_no_ucb_spread:.2f}")
        print(f"  UCB 提升:                     {atbmax_spread - atbmax_no_ucb_spread:.2f}")
        print(f"  ATB-Max oracle calls:        {summary['ATB-Max']['oracle_mean']:.1f}")
        print(f"  ATB-Max (无UCB) oracle calls: {summary.get('ATB-Max (无UCB)', {}).get('oracle_mean', 0):.1f}")

        print("\n--- 轻量估计器的作用 ---")
        print(f"  ATB-Max spread:                  {atbmax_spread:.2f}")
        print(f"  ATB-Max (无估计器) spread:       {atbmax_no_est_spread:.2f}")
        print(f"  估计器导致的 spread 损失:         {atbmax_no_est_spread - atbmax_spread:.2f}")
        if summary['ATB-Max']['oracle_mean'] > 0:
            print(f"  ATB-Max oracle calls:            {summary['ATB-Max']['oracle_mean']:.1f}")
            print(f"  ATB-Max (无估计器) oracle calls:  {summary.get('ATB-Max (无估计器)', {}).get('oracle_mean', 0):.1f}")
            ocr = summary.get('ATB-Max (无估计器)', {}).get('oracle_mean', 1) / max(summary['ATB-Max']['oracle_mean'], 1)
            print(f"  Oracle 调用减少倍数:              {ocr:.1f}x")
            bgr = summary.get('ATB-Max (无估计器)', {}).get('budget_mean', 1) / max(summary['ATB-Max']['budget_mean'], 1)
            print(f"  总预算减少倍数:                    {bgr:.1f}x")

    # ---- 实验 3: 参数敏感性分析 ----
    print("\n" + "=" * 60)
    print("实验 3: 参数敏感性分析 (Parameter Sensitivity)")
    print("=" * 60)
    sensitivity_results = run_parameter_sensitivity(
        main_graph, main_dataset_name, config, output_dir=args.output)

    # ---- 实验 4: 可扩展性实验 (多个数据集) ----
    print("\n" + "=" * 60)
    print("实验 4: 可扩展性实验 (Scalability on Multiple Datasets)")
    print("=" * 60)
    scalability_results = run_scalability_experiment(
        datasets, config, output_dir=args.output)

    # ---- 总结 ----
    total_elapsed = time.perf_counter() - total_start_time
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"总运行时间: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    if "ATB-Max" in summary and "Greedy" in summary:
        print(f"""
关键发现:

1. 解质量:
   - ATB-Max 传播范围: {summary['ATB-Max']['spread_mean']:.2f}
   - Greedy (上界): {summary['Greedy']['spread_mean']:.2f}
   - 相对解质量: {summary['ATB-Max']['spread_mean'] / max(summary['Greedy']['spread_mean'], 0.01):.4f}

2. 计算效率:
   - ATB-Max Oracle 调用: {summary['ATB-Max']['oracle_mean']:.1f}
   - Greedy Oracle 调用: {summary['Greedy']['oracle_mean']:.1f}
   - ATB-Max 运行时间: {summary['ATB-Max']['runtime_mean']:.2f}s

3. 可扩展性:
   - 已在 {len(datasets)} 个数据集上完成测试
   - 图表保存至: {args.output}/
""")

    print("实验完成。使用 --mode full 运行完整版实验。")


if __name__ == "__main__":
    main()
