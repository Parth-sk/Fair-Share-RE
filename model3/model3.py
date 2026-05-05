from mesa import Agent, Model
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER AGENT
# -----------------------------
class UserAgent(Agent):
    def __init__(self, unique_id, model, user_type):
        super().__init__(unique_id, model)

        self.user_type = user_type

        # --- Demand parameters (given: alpha fixed, beta now Gaussian) ---
        if user_type == "LTG":
            self.alpha = 10
            self.beta = max(0.01, np.random.normal(1.2, 0.3))
        else:
            self.alpha = 8
            self.beta = max(0.01, np.random.normal(1.5, 0.3))

        # --- Preference weights (4 cases, Gaussian) ---
        if user_type == "LTG":
            self.v_LTG = max(0.01, np.random.normal(1.2, 0.2))  # LTG user choosing LTG
            self.v_STG = max(0.01, np.random.normal(0.8, 0.2))  # LTG user choosing STG
        else:
            self.v_LTG = max(0.01, np.random.normal(0.8, 0.2))  # STG user choosing LTG
            self.v_STG = max(0.01, np.random.normal(1.2, 0.2))  # STG user choosing STG

        self.choice = user_type
    def step(self):

      # Demand
      q = max(0, self.alpha - self.beta * self.model.p_b)

      # Bandwidth payment
      P_b = self.model.p_b * q

      # Content price
      if self.model.levy_on and self.model.pass_through:
          p_c_LTG = self.model.p_c_LTG + self.model.p_l
      else:
          p_c_LTG = self.model.p_c_LTG

            # --- Normalization scale ---
      scale = p_c_LTG   # simpler, less aggressive

      # Normalize prices
      P_b_norm = P_b / scale
      p_c_LTG_norm = p_c_LTG / scale
      p_c_STG_norm = self.model.p_c_STG / scale

      # Utility (same formula, normalized prices)
      U_LTG = self.model.Q * self.v_LTG * q - (p_c_LTG_norm + P_b_norm)
      U_STG = self.model.Q * self.v_STG * q - (p_c_STG_norm + P_b_norm)

      # Choice
      if U_LTG >= U_STG:
          self.choice = "LTG"
      else:
          self.choice = "STG"

      self.q = q


# -----------------------------
# MODEL
# -----------------------------
class TelecomModel(Model):
    def __init__(self, N=1000, levy_on=False, pass_through=True):
        super().__init__()

        self.num_agents = N
        self.agents = []

        # Prices
        self.p_b = 1.0
        self.p_c_LTG = 4.0
        self.p_c_STG = 3.5
        self.p_l = 1

        # Network
        self.capacity = 12000

        # Policy toggles
        self.levy_on = levy_on
        self.pass_through = pass_through

        # Metrics
        self.total_traffic = 0
        self.Q = 1.0

        # Create agents
        for i in range(self.num_agents):
            if i < 0.75 * self.num_agents:
                user_type = "LTG"
            else:
                user_type = "STG"

            agent = UserAgent(i, self, user_type)
            self.agents.append(agent)

        self.datacollector = DataCollector(
          model_reporters={
              "Total Traffic": lambda m: m.total_traffic,
              "Quality": lambda m: m.Q,
              "LTG Users": lambda m: sum(1 for a in m.agents if a.choice == "LTG"),
              "STG Users": lambda m: sum(1 for a in m.agents if a.choice == "STG"),
              "LTG Traffic": lambda m: sum(a.q for a in m.agents if a.choice == "LTG"),
              "STG Traffic": lambda m: sum(a.q for a in m.agents if a.choice == "STG"),
              "LTG Revenue": lambda m: m.R_LTG,
              "STG Revenue": lambda m: m.R_STG,
          }
      )
    def step(self):

        # Compute total demand (before choice)
        total_q = sum(max(0, a.alpha - a.beta * self.p_b) for a in self.agents)

        # Compute quality
        self.Q = max(0, (self.capacity - total_q) / self.capacity)

        # Each agent decides
        for agent in self.agents:
            agent.step()

        # Aggregate traffic
        total_LTG = sum(a.q for a in self.agents if a.choice == "LTG")
        total_STG = sum(a.q for a in self.agents if a.choice == "STG")
        # Subscriber counts
        n_LTG = sum(1 for a in self.agents if a.choice == "LTG")
        n_STG = sum(1 for a in self.agents if a.choice == "STG")

        # LTG price (with levy if passed)
        if self.levy_on and self.pass_through:
            P_c_LTG = self.p_c_LTG + self.p_l
        else:
            P_c_LTG = self.p_c_LTG

        # STG price
        P_c_STG = self.p_c_STG

        # Revenues (STRICT equations)
        self.R_LTG = P_c_LTG * n_LTG - (1 if self.levy_on else 0) * self.p_l * total_LTG
        self.R_STG = P_c_STG * n_STG

        self.total_traffic = total_LTG + total_STG

        # Update quality again
        self.Q = max(0, (self.capacity - self.total_traffic) / self.capacity)

        # Collect data
        self.datacollector.collect(self)

   


# -----------------------------
# RUN SIMULATION
# -----------------------------
model = TelecomModel(N=1000, levy_on=False, pass_through=False)

# Before levy
for _ in range(200):
    model.step()

# Activate levy
model.levy_on = True

# After levy
for _ in range(200):
    model.step()

#After pass through
model.pass_through = True
for _ in range(200):
    model.step()


# -----------------------------
# RESULTS
# -----------------------------
data = model.datacollector.get_model_vars_dataframe()

# -------- EXISTING PLOTS --------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0,0].plot(data["Total Traffic"])
axs[0,0].set_title("Total Traffic")

axs[0,1].plot(data["Quality"])
axs[0,1].set_title("Quality")

axs[1,0].plot(data["LTG Users"], label="LTG")
axs[1,0].plot(data["STG Users"], label="STG")
axs[1,0].legend()
axs[1,0].set_title("User Choice")

axs[1,1].plot(data["LTG Traffic"], label="LTG")
axs[1,1].plot(data["STG Traffic"], label="STG")
axs[1,1].legend()
axs[1,1].set_title("Traffic Split")

plt.tight_layout()
plt.show()


# -------- ADD THIS HERE --------
plt.figure(figsize=(8, 5))
plt.plot(data["LTG Revenue"], label="LTG Revenue")
plt.plot(data["STG Revenue"], label="STG Revenue")
plt.legend()
plt.title("OTT Revenues")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.show()