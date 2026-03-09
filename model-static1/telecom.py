from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt


class UserAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.platform = None
        self.q = 0
        self.utility = 0

    def choose_platform(self):
        # 75% probability choose LTG
        if random.random() < 0.75:
            self.platform = "LTG"
        else:
            self.platform = "STG"

    def compute_demand(self):
        if self.platform == "LTG":
            Pc = self.model.Pc_LTG
        else:
            Pc = self.model.Pc_STG

        Pb = self.model.Pb

        q = self.model.alpha - self.model.beta * Pc - self.model.gamma * Pb
        self.q = max(q, 0)

        self.utility = self.model.v * self.q - (Pc + Pb)

    def step(self):
        self.choose_platform()
        self.compute_demand()


class TelecomModel(Model):
    def __init__(self, N=200):
        self.num_agents = N
        self.schedule = RandomActivation(self)

        # PARAMETERS
        self.alpha = 50
        self.beta = 0.05
        self.gamma = 0.02
        self.v = 30

        self.Pb = 500
        self.Pc_LTG = 300
        self.Pc_STG = 150
        self.Pt = 2

        self.K = 5000

        self.telecom_revenue = 0
        self.ltg_revenue = 0
        self.stg_revenue = 0
        self.avg_utility = 0

        for i in range(self.num_agents):
            agent = UserAgent(i, self)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Telecom Revenue": "telecom_revenue",
                "LTG Revenue": "ltg_revenue",
                "STG Revenue": "stg_revenue",
                "Average Utility": "avg_utility"
            }
        )

    def compute_revenues(self):
        total_q_ltg = 0
        total_q_stg = 0
        ltg_users = 0
        stg_users = 0
        total_utility = 0

        for agent in self.schedule.agents:
            total_utility += agent.utility

            if agent.platform == "LTG":
                total_q_ltg += agent.q
                ltg_users += 1
            else:
                total_q_stg += agent.q
                stg_users += 1

        # Telecom revenue
        self.telecom_revenue = self.Pb * self.num_agents + self.Pt * total_q_ltg

        # LTG revenue
        self.ltg_revenue = ltg_users * self.Pc_LTG - self.Pt * total_q_ltg

        # STG revenue
        self.stg_revenue = stg_users * self.Pc_STG

        self.avg_utility = total_utility / self.num_agents

    def step(self):
        self.schedule.step()
        self.compute_revenues()
        self.datacollector.collect(self)


# RUN SIMULATION
model = TelecomModel(N=2000)

steps = 50
for i in range(steps):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

plt.figure()
plt.plot(data["Telecom Revenue"], label="Telecom Revenue")
plt.plot(data["LTG Revenue"], label="LTG Revenue")
plt.plot(data["STG Revenue"], label="STG Revenue")
plt.plot(data["Average Utility"], label="Average Utility")
plt.legend()
plt.xlabel("Step")
plt.show()