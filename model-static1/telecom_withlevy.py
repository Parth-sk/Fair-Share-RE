import tkinter as tk
from mesa import Agent, Model
from mesa.time import RandomActivation
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# --------------------
# USER AGENT
# --------------------
class UserAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.q = 0
        self.utility = 0

        #75% choose ltg
        if random.random() < 0.75:
            self.platform = "LTG"
        else:
            self.platform = "STG"

    def step(self):

        if self.platform == "LTG":
            Pc = self.model.Pc_LTG
        else:
            Pc = self.model.Pc_STG

        Pb = self.model.Pb

        q = self.model.alpha - self.model.beta * Pc - self.model.gamma * Pb
        self.q = max(q, 0)

        self.utility = self.model.v * self.q - (Pc + Pb)
        

# --------------------
# MODEL
# --------------------
class TelecomModel(Model):
    def __init__(self, N=200):
        self.num_agents = N
        self.schedule = RandomActivation(self)

        # Parameters
        self.alpha = 50
        self.beta = 0.05
        self.gamma = 0.02
        self.v = 30

        self.Pb = 500
        self.Pc_LTG = 300
        self.Pc_STG = 150
        self.Pt = 2  # levy rate

        self.levy_active = False

        self.telecom_revenue = 0
        self.ltg_revenue = 0
        self.stg_revenue = 0
        self.avg_utility = 0

        for i in range(self.num_agents):
            self.schedule.add(UserAgent(i, self))

    def step(self):
        self.schedule.step()

        total_q_ltg = 0
        ltg_users = 0
        stg_users = 0
        total_utility = 0

        for agent in self.schedule.agents:
            total_utility += agent.utility
            if agent.platform == "LTG":
                total_q_ltg += agent.q
                ltg_users += 1
            else:
                stg_users += 1

        if self.levy_active:
            levy_amount = self.Pt * total_q_ltg
        else:
            levy_amount = 0

        self.telecom_revenue = self.Pb * self.num_agents + levy_amount
        self.ltg_revenue = ltg_users * self.Pc_LTG - levy_amount
        self.stg_revenue = stg_users * self.Pc_STG
        self.avg_utility = total_utility / self.num_agents


# --------------------
# GUI
# --------------------
class App:
    def __init__(self, root):
        self.model = TelecomModel(2000)
        self.root = root
        self.root.title("Telecom Market Simulation")
        self.root.geometry("900x600")

        # Allow resizing
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Frame to hold plot
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0, sticky="nsew")

        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Control frame
        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=1, column=0, sticky="ew")

        # Buttons
        self.step_button = tk.Button(self.control_frame, text="Run Step", command=self.run_step)
        self.step_button.pack(side="left", padx=10, pady=5)

        self.levy_button = tk.Button(self.control_frame, text="Activate Levy", command=self.toggle_levy)
        self.levy_button.pack(side="left", padx=10, pady=5)

        # Data storage
        self.telecom_data = []
        self.ltg_data = []
        self.stg_data = []
        self.utility_data = []


        self.line1, = self.ax.plot([], [], label="Telecom Revenue")
        self.line2, = self.ax.plot([], [], label="LTG Revenue")
        self.line3, = self.ax.plot([], [], label="STG Revenue")
        self.line4, = self.ax.plot([], [], label="Avg Utility")

        self.ax.legend(loc="upper left")

    def toggle_levy(self):
        self.model.levy_active = not self.model.levy_active
        if self.model.levy_active:
            self.levy_button.config(text="Deactivate Levy")
        else:
            self.levy_button.config(text="Activate Levy")

    def run_step(self):
        self.model.step()

        self.telecom_data.append(self.model.telecom_revenue)
        self.ltg_data.append(self.model.ltg_revenue)
        self.stg_data.append(self.model.stg_revenue)
        self.utility_data.append(self.model.avg_utility)

        self.line1.set_data(range(len(self.telecom_data)), self.telecom_data)
        self.line2.set_data(range(len(self.ltg_data)), self.ltg_data)
        self.line3.set_data(range(len(self.stg_data)), self.stg_data)
        self.line4.set_data(range(len(self.utility_data)), self.utility_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()


# --------------------
# RUN APP
# --------------------
root = tk.Tk()
app = App(root)
root.mainloop()