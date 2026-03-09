import tkinter as tk
from mesa import Agent, Model
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class JobSeeker(Agent):
    def __init__(self, unique_id, skills, rating, matchrate, ask_rate):
        self.unique_id = unique_id
        self.skills = skills
        self.rating = rating
        self.matchrate = matchrate
        self.ask_rate = ask_rate
        self.is_employed = False
        self.accepted_rate = None
        self.x, self.y = random.randint(50, 300), random.randint(50, 850)  # Random initial position


class Vacancy(Agent):
    def __init__(self, unique_id, category, offer_rate):
        self.unique_id = unique_id
        self.category = category
        self.offer_rate = offer_rate
        self.is_filled = False
        self.applicants = []
        self.x, self.y = random.randint(400, 650), random.randint(50, 850)  # Random initial position

class JobMatchingModel(Model):
    def __init__(self, num_job_seekers, num_vacancies, ask_rate_range, offer_rate_range):
        super().__init__()
        self.job_seekers = []
        self.vacancies = []
        self.matched_counts = []
        self.accepted_rates = []
        self.ask_offer_rates = []  # ✅ Fix: Added this missing attribute

        # Create job seekers
        for i in range(num_job_seekers):
            skills = random.sample([
                "data-mining", "data-processing", "data-visualization",
                "deep-learning", "machine-learning", "data-analytics",
                "data-engineering", "data-extraction"
            ], random.randint(1, 8))
            rating = random.randint(0, 5)
            matchrate = random.uniform(0, 1)
            ask_rate = random.randint(*ask_rate_range)
            job_seeker = JobSeeker(i, skills, rating, matchrate, ask_rate)
            self.job_seekers.append(job_seeker)

        # Create vacancies
        for i in range(num_vacancies):
            category = random.choice([
                "data-mining", "data-processing", "data-visualization",
                "deep-learning", "machine-learning", "data-analytics",
                "data-engineering", "data-extraction"
            ])
            offer_rate = random.randint(*offer_rate_range)
            vacancy = Vacancy(num_job_seekers + i, category, offer_rate)
            self.vacancies.append(vacancy)

    def step(self, step_number):
        # Reset applicants for vacancies
        for vacancy in self.vacancies:
            vacancy.applicants = []

        # Job seekers apply to vacancies
        for job_seeker in self.job_seekers:
            if not job_seeker.is_employed:
                available_vacancies = [v for v in self.vacancies if not v.is_filled]
                if available_vacancies:
                    chosen_vacancy = random.choice(available_vacancies)
                    chosen_vacancy.applicants.append(job_seeker)

        # Vacancy selects the best applicant
        for vacancy in self.vacancies:
            if vacancy.is_filled or not vacancy.applicants:
                continue

            best_applicant = max(
                (a for a in vacancy.applicants if vacancy.category in a.skills),
                key=lambda a: a.matchrate + a.rating,
                default=None
            )

            if best_applicant:
                best_applicant.is_employed = True
                vacancy.is_filled = True
                accepted_rate = (best_applicant.ask_rate + vacancy.offer_rate) / 2
                best_applicant.accepted_rate = accepted_rate
                self.accepted_rates.append(accepted_rate)
                best_applicant.x, best_applicant.y = vacancy.x - 50, vacancy.y  # Move next to matched vacancy

                # ✅ Fix: Track ask & offer rates over time
                self.ask_offer_rates.append((best_applicant.ask_rate, vacancy.offer_rate))

        # Record counts for plotting
        matched_job_seekers = sum(1 for js in self.job_seekers if js.is_employed)
        unmatched_job_seekers = len(self.job_seekers) - matched_job_seekers
        matched_vacancies = sum(1 for v in self.vacancies if v.is_filled)
        unmatched_vacancies = len(self.vacancies) - matched_vacancies

        self.matched_counts.append((step_number, matched_job_seekers, unmatched_job_seekers, matched_vacancies, unmatched_vacancies))

class JobMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Matching Simulation")

        # Main frame layout (16:9 aspect ratio)
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (plots)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(6, 9))  # Three stacked plots
        self.chart_canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel (canvas for agent visualization)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, width=600, height=500, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls at the bottom
        controls_frame = tk.Frame(root)
        controls_frame.pack()

        self.num_job_seekers_var = tk.StringVar(value="85")
        self.num_vacancies_var = tk.StringVar(value="100")
        self.ask_rate_min_var = tk.StringVar(value="500")
        self.ask_rate_max_var = tk.StringVar(value="1500")
        self.offer_rate_min_var = tk.StringVar(value="300")
        self.offer_rate_max_var = tk.StringVar(value="1200")

        tk.Label(controls_frame, text="Job Seekers:").grid(row=0, column=0)
        self.num_job_seekers_entry = tk.Entry(controls_frame, textvariable=self.num_job_seekers_var)
        self.num_job_seekers_entry.grid(row=0, column=1)

        tk.Label(controls_frame, text="Vacancies:").grid(row=1, column=0)
        self.num_vacancies_entry = tk.Entry(controls_frame, textvariable=self.num_vacancies_var)
        self.num_vacancies_entry.grid(row=1, column=1)

        tk.Label(controls_frame, text="Ask Rate Range (min, max):").grid(row=2, column=0)
        self.ask_rate_min_entry = tk.Entry(controls_frame, textvariable=self.ask_rate_min_var)
        self.ask_rate_min_entry.grid(row=2, column=1)
        self.ask_rate_max_entry = tk.Entry(controls_frame, textvariable=self.ask_rate_max_var)
        self.ask_rate_max_entry.grid(row=2, column=2)

        tk.Label(controls_frame, text="Offer Rate Range (min, max):").grid(row=3, column=0)
        self.offer_rate_min_entry = tk.Entry(controls_frame, textvariable=self.offer_rate_min_var)
        self.offer_rate_min_entry.grid(row=3, column=1)
        self.offer_rate_max_entry = tk.Entry(controls_frame, textvariable=self.offer_rate_max_var)
        self.offer_rate_max_entry.grid(row=3, column=2)

        self.step_button = tk.Button(controls_frame, text="Step", command=self.run_step)
        self.step_button.grid(row=4, column=0, columnspan=3)

        # Model instance
        self.model = None

    def initialize_model(self):
        num_job_seekers = int(self.num_job_seekers_entry.get() or 85)
        num_vacancies = int(self.num_vacancies_entry.get() or 100)
        ask_rate_range = (int(self.ask_rate_min_entry.get() or 500), int(self.ask_rate_max_entry.get() or 1500))
        offer_rate_range = (int(self.offer_rate_min_entry.get() or 300), int(self.offer_rate_max_entry.get() or 1200))
        self.model = JobMatchingModel(num_job_seekers, num_vacancies, ask_rate_range, offer_rate_range)

    def draw_agents(self):
        self.canvas.delete("all")

        for job_seeker in self.model.job_seekers:
            color = "blue" if job_seeker.is_employed else "orange"
            self.canvas.create_oval(
                job_seeker.x, job_seeker.y, job_seeker.x + 20, job_seeker.y + 20, fill=color
            )
            self.canvas.create_text(job_seeker.x + 10, job_seeker.y + 30, text=f"JS{job_seeker.unique_id}: {job_seeker.ask_rate}")
        
        for vacancy in self.model.vacancies:
            color = "green" if vacancy.is_filled else "red"
            self.canvas.create_rectangle(
                vacancy.x, vacancy.y, vacancy.x + 30, vacancy.y + 20, fill=color
            )
            self.canvas.create_text(vacancy.x + 15, vacancy.y + 30, text=f"V{vacancy.unique_id}: {vacancy.offer_rate}")
            
        for job_seeker in self.model.job_seekers:
            if job_seeker.is_employed:
                self.canvas.create_text(
                    job_seeker.x + 40, job_seeker.y + 10, 
                    text=f"JS{job_seeker.unique_id} ↔ V{job_seeker.unique_id+10} \n {job_seeker.accepted_rate}"
                )

    def run_step(self):
        if not self.model:
            self.initialize_model()

        step_number = len(self.model.matched_counts)
        self.model.step(step_number)

        self.draw_agents()
        self.update_plots()

    def update_plots(self):
        steps = [count[0] for count in self.model.matched_counts]
        matched_job_seekers = [count[1] for count in self.model.matched_counts]
        unmatched_job_seekers = [count[2] for count in self.model.matched_counts]
        matched_vacancies = [count[3] for count in self.model.matched_counts]
        unmatched_vacancies = [count[4] for count in self.model.matched_counts]

        # Plot 1: Job Matching Process
        self.axs[0].clear()
        self.axs[0].plot(steps, matched_job_seekers, marker='o', label="Matched Job Seekers")
        self.axs[0].plot(steps, unmatched_job_seekers, marker='o', label="Unmatched Job Seekers")
        self.axs[0].plot(steps, matched_vacancies, marker='o', label="Matched Vacancies")
        self.axs[0].plot(steps, unmatched_vacancies, marker='o', label="Unmatched Vacancies")
        self.axs[0].set_title("Job Matching Process")
        self.axs[0].legend()

        # Plot 2: Accepted Rates Over Time
        steps_rates = list(range(len(self.model.accepted_rates)))
        self.axs[1].clear()
        self.axs[1].plot(steps_rates, self.model.accepted_rates, marker='o', label="Accepted Rates")
        self.axs[1].set_title("Accepted Rates Over Time")
        self.axs[1].legend()

        # Plot 3: Ask vs Offer Rates
        ask_rates = [rate[0] for rate in self.model.ask_offer_rates]
        offer_rates = [rate[1] for rate in self.model.ask_offer_rates]
        self.axs[2].clear()
        self.axs[2].plot(steps_rates, ask_rates, marker='o', label="Ask Rates")
        self.axs[2].plot(steps_rates, offer_rates, marker='o', label="Offer Rates")
        self.axs[2].set_title("Ask vs Offer Rates")
        self.axs[2].legend()

        self.chart_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = JobMatchingApp(root)
    root.mainloop()
