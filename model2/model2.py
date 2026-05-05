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

        # 75% LTG preferred
        if random.random() < 0.75:
            self.type = "LTG_pref"
        else:
            self.type = "STG_pref"

        self.platform = "LTG" if self.type == "LTG_pref" else "STG"

        self.q = 0
        self.utility = 0

    def demand(self):
        """
        q_c^n = alpha_i - beta_i * p_b
        """
        if self.type == "LTG_pref":
            alpha = self.model.alpha_LTG_pref
        else:
            alpha = self.model.alpha_STG_pref

        q = alpha - self.model.beta * self.model.p_b
        return max(q, 0)

    def utility_if_choose(self, platform):

        q = self.demand()

        # content price
        if platform == "LTG":
            Pc = self.model.Pc_LTG_current
        else:
            Pc = self.model.p_c_STG

        Pb = self.model.p_b * q

        # preference weight
        if self.type == "LTG_pref":

            if platform == "LTG":
                v = self.model.v_LTG_pref_LTG
            else:
                v = self.model.v_LTG_pref_STG

        else:

            if platform == "LTG":
                v = self.model.v_STG_pref_LTG
            else:
                v = self.model.v_STG_pref_STG

        Q = self.model.quality

        U = Q * v * q - (Pc + Pb)

        return U, q

    def step(self):

        U_ltg, q_ltg = self.utility_if_choose("LTG")
        U_stg, q_stg = self.utility_if_choose("STG")

        # choose best option
        if U_ltg > U_stg:
            self.platform = "LTG"
            self.utility = U_ltg
            self.q = q_ltg

        else:
            self.platform = "STG"
            self.utility = U_stg
            self.q = q_stg


# --------------------
# MODEL
# --------------------
class TelecomModel(Model):

    def __init__(self, N=2000):

        self.num_agents = N
        self.schedule = RandomActivation(self)

        # -------- parameters --------

        self.alpha_LTG_pref = 12
        self.alpha_STG_pref = 10

        self.beta = 0.6

        self.v_LTG_pref_LTG = 1.2
        self.v_LTG_pref_STG = 0.8

        self.v_STG_pref_LTG = 0.8
        self.v_STG_pref_STG = 1.2

        self.p_c_LTG = 5
        self.p_c_STG = 4

        self.p_b = 1.0

        self.levy = 1.0

        self.capacity = 30000

        # dynamic state

        self.levy_active = False
        self.ltg_passes_levy = False

        self.quality = 1

        self.total_traffic = 0

        self.telecom_revenue = 0
        self.ltg_revenue = 0
        self.stg_revenue = 0
        self.avg_utility = 0

        self.market_share_ltg = 0

        self.time = 0

        # create agents
        for i in range(self.num_agents):

            agent = UserAgent(i, self)
            self.schedule.add(agent)

    @property
    def Pc_LTG_current(self):

        if self.levy_active and self.ltg_passes_levy:

            return self.p_c_LTG + self.levy

        return self.p_c_LTG

    def compute_quality(self):

        C = self.total_traffic

        Q = (self.capacity - C) / self.capacity

        return max(Q, 0)

    def compute_revenues(self, q_ltg, n_ltg, n_stg):

        levy_payment = 0

        if self.levy_active:

            levy_payment = self.levy * q_ltg

        telecom_rev = self.p_b * self.total_traffic + levy_payment

        ltg_rev = self.Pc_LTG_current * n_ltg - levy_payment

        stg_rev = self.p_c_STG * n_stg

        return telecom_rev, ltg_rev, stg_rev

    def step(self):

        self.time += 1

        # users choose OTT
        self.schedule.step()

        # aggregate results
        total_q = 0
        q_ltg = 0

        n_ltg = 0
        n_stg = 0

        total_utility = 0

        for agent in self.schedule.agents:

            total_q += agent.q
            total_utility += agent.utility

            if agent.platform == "LTG":

                q_ltg += agent.q
                n_ltg += 1

            else:

                n_stg += 1

        self.total_traffic = total_q

        self.quality = self.compute_quality()

        telecom_rev, ltg_rev, stg_rev = self.compute_revenues(q_ltg, n_ltg, n_stg)

        self.telecom_revenue = telecom_rev
        self.ltg_revenue = ltg_rev
        self.stg_revenue = stg_rev

        self.avg_utility = total_utility / self.num_agents

        self.market_share_ltg = n_ltg / self.num_agents

        print(
            f"Step {self.time} | "
            f"LTG: {n_ltg} ({n_ltg/self.num_agents:.2f}) | "
            f"STG: {n_stg} ({n_stg/self.num_agents:.2f}) | "
            f"Q: {self.quality:.3f}"
        )


# --------------------
# GUI
# --------------------
class App:

    def __init__(self, root):

        self.model = TelecomModel()

        self.root = root
        self.root.title("Telecom Levy Simulation")

        self.fig, self.ax = plt.subplots(2, 1, figsize=(7, 8))

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # buttons
        button_frame = tk.Frame(root)
        button_frame.pack()

        tk.Button(
            button_frame,
            text="Run Step",
            command=self.run_step
        ).pack(side="left")

        tk.Button(
            button_frame,
            text="Activate Levy",
            command=self.toggle_levy
        ).pack(side="left")

        tk.Button(
            button_frame,
            text="Toggle Pass-through",
            command=self.toggle_passthrough
        ).pack(side="left")

        # data storage

        self.telecom_data = []
        self.ltg_data = []
        self.stg_data = []
        self.utility_data = []

        self.market_share_data = []
        self.traffic_data = []
        self.quality_data = []

        # plots

        self.line1, = self.ax[0].plot([], [], label="Telecom revenue")
        self.line2, = self.ax[0].plot([], [], label="LTG revenue")
        self.line3, = self.ax[0].plot([], [], label="STG revenue")
        self.line4, = self.ax[0].plot([], [], label="Avg utility")

        self.line5, = self.ax[1].plot([], [], label="LTG share")
        self.line6, = self.ax[1].plot([], [], label="Total traffic")
        self.line7, = self.ax[1].plot([], [], label="Quality")

        self.ax[0].legend()
        self.ax[1].legend()

    def toggle_levy(self):

        self.model.levy_active = not self.model.levy_active

    def toggle_passthrough(self):

        self.model.ltg_passes_levy = not self.model.ltg_passes_levy

    def run_step(self):

        self.model.step()

        # store data
        self.telecom_data.append(self.model.telecom_revenue)
        self.ltg_data.append(self.model.ltg_revenue)
        self.stg_data.append(self.model.stg_revenue)
        self.utility_data.append(self.model.avg_utility)

        self.market_share_data.append(self.model.market_share_ltg)
        self.traffic_data.append(self.model.total_traffic)
        self.quality_data.append(self.model.quality)

        t = range(len(self.telecom_data))

        # update plots

        self.line1.set_data(t, self.telecom_data)
        self.line2.set_data(t, self.ltg_data)
        self.line3.set_data(t, self.stg_data)
        self.line4.set_data(t, self.utility_data)

        self.line5.set_data(t, self.market_share_data)
        self.line6.set_data(t, self.traffic_data)
        self.line7.set_data(t, self.quality_data)

        for axis in self.ax:

            axis.relim()
            axis.autoscale_view()

        self.canvas.draw()
        
    def on_close(self):
        self.root.quit()      # stops mainloop
        self.root.destroy()   # destroys window
        exit()                # ensures full process termination


# --------------------
# RUN
# --------------------

root = tk.Tk()
app = App(root)

root.mainloop()
