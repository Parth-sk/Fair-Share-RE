"""
Model 1: Static Baseline ABM
1 ISP, 2 LTGs, 1 User population
Mesa-based, single-step equilibrium
"""

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# AGENTS
# ==========================================================

class UserAgent(Agent):
    """
    Aggregate user population
    """
    def __init__(self, unique_id, model, N, avg_demand, p_LTG_A):
        super().__init__(unique_id, model)
        self.N = N
        self.avg_demand = avg_demand
        self.p_LTG_A = p_LTG_A

    def step(self):
        total_traffic = self.N * self.avg_demand
        self.model.total_traffic = total_traffic
        self.model.traffic_A = self.p_LTG_A * total_traffic
        self.model.traffic_B = (1 - self.p_LTG_A) * total_traffic


class LTGAgent(Agent):
    """
    Large Traffic Generator (static)
    """
    def __init__(self, unique_id, model, name):
        super().__init__(unique_id, model)
        self.name = name
        self.traffic = 0.0

    def step(self):
        if self.name == "A":
            self.traffic = self.model.traffic_A
        else:
            self.traffic = self.model.traffic_B


class ISPAgent(Agent):
    """
    ISP accounting + congestion
    """
    def __init__(self, unique_id, model, capacity_gbps, arpu,
                 transit_cost_per_gb, fixed_opex):
        super().__init__(unique_id, model)
        self.capacity_gbps = capacity_gbps
        self.arpu = arpu
        self.transit_cost_per_gb = transit_cost_per_gb
        self.fixed_opex = fixed_opex

        self.utilization = 0.0
        self.profit = 0.0

    def step(self):
        # Convert GB/day -> Gbps
        load_gbps = (self.model.total_traffic * 8) / 86400
        self.utilization = load_gbps / self.capacity_gbps

        revenue = self.model.users.N * self.arpu
        transit_cost = (
            self.model.total_traffic * 30 * self.transit_cost_per_gb
        )

        self.profit = revenue - transit_cost - self.fixed_opex


# ==========================================================
# MODEL
# ==========================================================

class BaselineModel(Model):
    """
    Static baseline model (single equilibrium step)
    """

    def __init__(
        self,
        N_users=50000,
        avg_demand=2.5,
        p_LTG_A=0.6,
        capacity_gbps=60,
        arpu=220,
        transit_cost=0.6,
        fixed_opex=3_000_000
    ):
        super().__init__()
        self.schedule = BaseScheduler(self)

        # Shared traffic state
        self.total_traffic = 0.0
        self.traffic_A = 0.0
        self.traffic_B = 0.0

        # Agents
        self.users = UserAgent(
            unique_id=0,
            model=self,
            N=N_users,
            avg_demand=avg_demand,
            p_LTG_A=p_LTG_A
        )

        self.ltg_A = LTGAgent(1, self, "A")
        self.ltg_B = LTGAgent(2, self, "B")

        self.isp = ISPAgent(
            unique_id=3,
            model=self,
            capacity_gbps=capacity_gbps,
            arpu=arpu,
            transit_cost_per_gb=transit_cost,
            fixed_opex=fixed_opex
        )

        for agent in [self.users, self.ltg_A, self.ltg_B, self.isp]:
            self.schedule.add(agent)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "TotalTraffic_GB_per_day": lambda m: m.total_traffic,
                "Traffic_LTG_A": lambda m: m.ltg_A.traffic,
                "Traffic_LTG_B": lambda m: m.ltg_B.traffic,
                "ISP_Utilization": lambda m: m.isp.utilization,
                "ISP_Profit": lambda m: m.isp.profit,
            }
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


# ==========================================================
# RUN MODEL
# ==========================================================

if __name__ == "__main__":

    model = BaselineModel()
    model.step()

    results = model.datacollector.get_model_vars_dataframe()
    print("\n=== BASELINE EQUILIBRIUM ===")
    print(results)

    # ======================================================
    # PLOTTING (Mesa-style)
    # ======================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Traffic split
    axes[0].bar(
        ["LTG A", "LTG B"],
        [results["Traffic_LTG_A"][0], results["Traffic_LTG_B"][0]]
    )
    axes[0].set_title("Traffic Split (GB/day)")
    axes[0].set_ylabel("GB/day")

    # Utilization
    axes[1].bar(
        ["ISP Utilization"],
        [results["ISP_Utilization"][0]]
    )
    axes[1].axhline(1.0, linestyle="--")
    axes[1].set_title("ISP Utilization")
    axes[1].set_ylim(0, max(1.2, results["ISP_Utilization"][0] * 1.1))

    # Profit
    axes[2].bar(
        ["ISP Profit"],
        [results["ISP_Profit"][0]]
    )
    axes[2].set_title("ISP Monthly Profit")
    axes[2].set_ylabel("₹")

    plt.tight_layout()
    plt.show()