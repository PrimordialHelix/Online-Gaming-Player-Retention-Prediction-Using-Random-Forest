"""
========================================================================================
🎮 APEX RETENTION AI • NATIVE DESKTOP APPLICATION
Modern Desktop GUI for Online Gaming Player Retention Prediction
Framework: CustomTkinter (Dark Mode) + Matplotlib TkAgg
Theme: Emerald Green (#10b981), Metallic Gold (#d4af37), Obsidian Black (#070b09)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Appearance settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Emerald & Gold Color Palette
COLOR_BG_DARK = "#070b09"
COLOR_CARD = "#0d1511"
COLOR_CARD_HOVER = "#131f19"
COLOR_EMERALD = "#10b981"
COLOR_EMERALD_HOVER = "#059669"
COLOR_EMERALD_LIGHT = "#34d399"
COLOR_GOLD = "#d4af37"
COLOR_GOLD_HOVER = "#b45309"
COLOR_GOLD_LIGHT = "#ffd700"
COLOR_TEXT_WHITE = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_ALERT_RED = "#ef4444"

class ApexRetentionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ApexRetention AI • Gaming Player Retention Platform")
        self.geometry("1300x820")
        self.minsize(1100, 720)
        self.configure(fg_color=COLOR_BG_DARK)

        # Train model pipeline
        self.load_and_train_model()

        # Build GUI Layout
        self.create_sidebar()
        self.create_main_content()

        # Initial prediction
        self.run_prediction()

    def load_and_train_model(self):
        """Loads dataset and fits the ColumnTransformer + RandomForest pipeline."""
        csv_path = "gaming_player_retention_dataset.csv"
        if not os.path.exists(csv_path):
            alt = os.path.join(os.path.dirname(__file__), "gaming_player_retention_dataset.csv")
            if os.path.exists(alt):
                csv_path = alt
            else:
                raise FileNotFoundError(f"Dataset {csv_path} not found!")

        self.df = pd.read_csv(csv_path)

        self.num_cols = [
            'age', 'playtime_hours', 'sessions_per_week', 
            'avg_session_duration', 'days_since_last_login', 
            'days_active_last_30', 'in_game_purchases', 'level_reached'
        ]
        self.cat_cols = ['game_genre']

        X = self.df.drop(columns=['player_id', 'retained'])
        y = self.df['retained']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), self.num_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        )

        self.pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('rf', rf)
        ])

        self.pipeline.fit(X_train, y_train)

        # Metrics
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        self.acc = accuracy_score(y_test, y_pred)
        self.prec = precision_score(y_test, y_pred)
        self.rec = recall_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_prob)

        # Retained cohort averages for radar comparison
        self.retained_means = self.df[self.df['retained'] == 1][
            ['playtime_hours', 'sessions_per_week', 'avg_session_duration', 'days_active_last_30', 'level_reached', 'in_game_purchases']
        ].mean()
        self.max_vals = self.df[
            ['playtime_hours', 'sessions_per_week', 'avg_session_duration', 'days_active_last_30', 'level_reached', 'in_game_purchases']
        ].max()

    def create_sidebar(self):
        """Constructs left sidebar with telemetry sliders and persona presets."""
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#060a08", border_width=1, border_color="#10b981")
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)

        # Header Title
        title_label = ctk.CTkLabel(
            self.sidebar_frame, text="🎮 APEX RETENTION AI",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_EMERALD_LIGHT
        )
        title_label.pack(padx=20, pady=(20, 4))

        sub_label = ctk.CTkLabel(
            self.sidebar_frame, text="Player Behavioral Telemetry HUD",
            font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED
        )
        sub_label.pack(padx=20, pady=(0, 15))

        # Quick Persona Presets
        presets_label = ctk.CTkLabel(
            self.sidebar_frame, text="⚡ QUICK PERSONA PRESETS",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_GOLD_LIGHT
        )
        presets_label.pack(padx=20, anchor="w", pady=(0, 5))

        preset_box = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        preset_box.pack(fill="x", padx=15, pady=2)

        btn_rpg = ctk.CTkButton(
            preset_box, text="🌟 RPG Veteran", width=135, height=28,
            fg_color="#064e3b", hover_color=COLOR_EMERALD, text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.load_preset(25, "RPG", 50, 9, 75, 1, 25, 4, 45)
        )
        btn_rpg.grid(row=0, column=0, padx=4, pady=3)

        btn_action = ctk.CTkButton(
            preset_box, text="⚡ Action Pro", width=135, height=28,
            fg_color="#064e3b", hover_color=COLOR_EMERALD, text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.load_preset(22, "Action", 35, 7, 50, 2, 18, 2, 30)
        )
        btn_action.grid(row=0, column=1, padx=4, pady=3)

        btn_casual = ctk.CTkButton(
            preset_box, text="⚠️ At-Risk Casual", width=135, height=28,
            fg_color="#78350f", hover_color=COLOR_GOLD, text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.load_preset(32, "Casual", 5, 2, 25, 7, 4, 0, 6)
        )
        btn_casual.grid(row=1, column=0, padx=4, pady=3)

        btn_churn = ctk.CTkButton(
            preset_box, text="🚨 Dormant Churn", width=135, height=28,
            fg_color="#7f1d1d", hover_color="#ef4444", text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.load_preset(38, "Strategy", 2, 1, 15, 18, 2, 0, 3)
        )
        btn_churn.grid(row=1, column=1, padx=4, pady=3)

        # Scrollable Telemetry Sliders
        self.slider_frame = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent")
        self.slider_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Controls
        self.genre_var = ctk.StringVar(value="RPG")
        self.create_option_menu(self.slider_frame, "Game Genre", self.genre_var, ['Action', 'RPG', 'Strategy', 'Shooter', 'Sports', 'Casual'])

        self.age_var = self.create_slider(self.slider_frame, "Player Age", 16, 60, 25, int)
        self.recency_var = self.create_slider(self.slider_frame, "Recency (Days Since Login)", 0, 28, 1, int, is_alert=True)
        self.active_days_var = self.create_slider(self.slider_frame, "Active Days (Past 30)", 1, 30, 24, int)
        self.sessions_var = self.create_slider(self.slider_frame, "Sessions Per Week", 1.0, 25.0, 8.5, float)
        self.duration_var = self.create_slider(self.slider_frame, "Avg Session Duration (Mins)", 15.0, 180.0, 72.0, float)
        self.playtime_var = self.create_slider(self.slider_frame, "Total Playtime (Hours)", 2.0, 120.0, 48.5, float)
        self.purchases_var = self.create_slider(self.slider_frame, "In-Game Purchases", 0, 20, 3, int)
        self.level_var = self.create_slider(self.slider_frame, "Level Reached (Progression)", 1, 80, 42, int)

    def create_slider(self, parent, label_text, min_val, max_val, default_val, val_type, is_alert=False):
        lbl_frame = ctk.CTkFrame(parent, fg_color="transparent")
        lbl_frame.pack(fill="x", pady=(8, 2))

        val_label = ctk.CTkLabel(
            lbl_frame, text=f"{default_val}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_GOLD_LIGHT if is_alert else COLOR_EMERALD_LIGHT
        )
        val_label.pack(side="right")

        title_lbl = ctk.CTkLabel(
            lbl_frame, text=label_text,
            font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_WHITE
        )
        title_lbl.pack(side="left")

        slider_var = ctk.DoubleVar(value=default_val)
        
        def on_change(value):
            formatted = val_type(value) if val_type is int else round(float(value), 1)
            val_label.configure(text=f"{formatted}")
            self.run_prediction()

        slider = ctk.CTkSlider(
            parent, from_=min_val, to=max_val, variable=slider_var,
            button_color=COLOR_EMERALD, button_hover_color=COLOR_GOLD,
            progress_color=COLOR_EMERALD, fg_color="#1e293b",
            command=on_change
        )
        slider.pack(fill="x", pady=(0, 6))
        return slider_var

    def create_option_menu(self, parent, label_text, var, options):
        lbl = ctk.CTkLabel(parent, text=label_text, font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_WHITE)
        lbl.pack(anchor="w", pady=(6, 2))

        menu = ctk.CTkOptionMenu(
            parent, values=options, variable=var,
            fg_color="#064e3b", button_color=COLOR_EMERALD,
            button_hover_color=COLOR_GOLD, dropdown_fg_color="#0d1511",
            dropdown_hover_color=COLOR_CARD_HOVER, dropdown_text_color="#ffffff",
            command=lambda _: self.run_prediction()
        )
        menu.pack(fill="x", pady=(0, 6))

    def load_preset(self, age, genre, playtime, sessions, duration, recency, active, purchases, level):
        self.age_var.set(age)
        self.genre_var.set(genre)
        self.playtime_var.set(playtime)
        self.sessions_var.set(sessions)
        self.duration_var.set(duration)
        self.recency_var.set(recency)
        self.active_days_var.set(active)
        self.purchases_var.set(purchases)
        self.level_var.set(level)
        self.run_prediction()

    def create_main_content(self):
        """Builds main content area with KPIs, prediction result, and charts."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(side="right", fill="both", expand=True, padx=20, pady=15)

        # Top Banner
        banner = ctk.CTkFrame(self.main_container, fg_color=COLOR_CARD, corner_radius=12, border_width=1, border_color="#10b981")
        banner.pack(fill="x", pady=(0, 15))

        banner_title = ctk.CTkLabel(
            banner, text="🎮 APEX RETENTION AI • LIVE-OPS PREDICTION DESKTOP",
            font=ctk.CTkFont(size=20, weight="bold"), text_color="#ffffff"
        )
        banner_title.pack(padx=20, pady=(12, 2), anchor="w")

        banner_sub = ctk.CTkLabel(
            banner, text="Random Forest Classifier (100 Trees) • 30-Day Pre-Cutoff Telemetry Analysis",
            font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_MUTED
        )
        banner_sub.pack(padx=20, pady=(0, 12), anchor="w")

        # KPI Metrics Row
        kpi_row = ctk.CTkFrame(self.main_container, fg_color="transparent")
        kpi_row.pack(fill="x", pady=(0, 15))

        self.add_kpi(kpi_row, "Accuracy", f"{self.acc*100:.1f}%", "Primary Metric", COLOR_EMERALD_LIGHT, 0)
        self.add_kpi(kpi_row, "Precision / Recall", f"{self.prec*100:.0f}% / {self.rec*100:.0f}%", f"F1: {self.f1:.3f}", COLOR_GOLD_LIGHT, 1)
        self.add_kpi(kpi_row, "ROC-AUC Score", f"{self.auc*100:.1f}%", "Discrimination", COLOR_EMERALD_LIGHT, 2)
        self.add_kpi(kpi_row, "Dataset Records", f"{len(self.df):,}", "Active Telemetry", "#ffffff", 3)

        # Split Container (Left: Prediction Result & Advice, Right: Radar Chart)
        middle_box = ctk.CTkFrame(self.main_container, fg_color="transparent")
        middle_box.pack(fill="both", expand=True)

        # Left Column: Result & Live-Ops Advice
        self.result_col = ctk.CTkFrame(middle_box, fg_color="transparent")
        self.result_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Dynamic Prediction Card
        self.card_result = ctk.CTkFrame(self.result_col, corner_radius=14, border_width=2, fg_color=COLOR_CARD)
        self.card_result.pack(fill="x", pady=(0, 15))

        self.res_badge = ctk.CTkLabel(
            self.card_result, text="STATUS: ACTIVE TELEMETRY",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_EMERALD_LIGHT
        )
        self.res_badge.pack(padx=20, pady=(16, 4))

        self.res_outcome = ctk.CTkLabel(
            self.card_result, text="🛡️ RETAINED (WILL RETURN)",
            font=ctk.CTkFont(size=24, weight="bold"), text_color="#ffffff"
        )
        self.res_outcome.pack(padx=20, pady=2)

        self.res_prob_lbl = ctk.CTkLabel(
            self.card_result, text="Retention Probability: 94.2%  |  Churn Risk: 5.8%",
            font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_MUTED
        )
        self.res_prob_lbl.pack(padx=20, pady=(0, 12))

        # Progress bar
        self.prob_bar = ctk.CTkProgressBar(self.card_result, height=14, corner_radius=7, progress_color=COLOR_EMERALD)
        self.prob_bar.pack(fill="x", padx=30, pady=(0, 18))

        # Tactical Live-Ops Advice Box
        self.card_advice = ctk.CTkFrame(self.result_col, corner_radius=12, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_GOLD)
        self.card_advice.pack(fill="both", expand=True)

        advice_title = ctk.CTkLabel(
            self.card_advice, text="⚡ LIVE-OPS TACTICAL INTERVENTION PLAN",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_GOLD_LIGHT
        )
        advice_title.pack(padx=16, pady=(12, 6), anchor="w")

        self.advice_text = ctk.CTkLabel(
            self.card_advice, text="", justify="left", wraplength=480,
            font=ctk.CTkFont(size=11), text_color="#e2e8f0"
        )
        self.advice_text.pack(padx=16, pady=(0, 12), anchor="w", fill="both", expand=True)

        # Right Column: Embedded Radar Chart
        self.chart_col = ctk.CTkFrame(middle_box, fg_color=COLOR_CARD, corner_radius=14, border_width=1, border_color="#10b981")
        self.chart_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        chart_hdr = ctk.CTkLabel(
            self.chart_col, text="📊 TELEMETRY BENCHMARK COMPARISON",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_GOLD_LIGHT
        )
        chart_hdr.pack(padx=15, pady=(12, 4), anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(5, 4.2), subplot_kw=dict(polar=True))
        self.fig.patch.set_facecolor(COLOR_CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_col)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def add_kpi(self, parent, title, val, sub, color, col_idx):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=10, border_width=1, border_color="#10b981")
        card.pack(side="left", fill="both", expand=True, padx=4)

        t_lbl = ctk.CTkLabel(card, text=title.upper(), font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_TEXT_MUTED)
        t_lbl.pack(padx=12, pady=(10, 2), anchor="w")

        v_lbl = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=20, weight="bold"), text_color=color)
        v_lbl.pack(padx=12, pady=0, anchor="w")

        s_lbl = ctk.CTkLabel(card, text=sub, font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_MUTED)
        s_lbl.pack(padx=12, pady=(0, 10), anchor="w")

    def run_prediction(self):
        """Constructs profile dict, runs through pipeline, and updates GUI and radar chart."""
        prof = {
            'age': float(self.age_var.get()),
            'game_genre': self.genre_var.get(),
            'playtime_hours': float(self.playtime_var.get()),
            'sessions_per_week': float(self.sessions_var.get()),
            'avg_session_duration': float(self.duration_var.get()),
            'days_since_last_login': int(self.recency_var.get()),
            'days_active_last_30': int(self.active_days_var.get()),
            'in_game_purchases': int(self.purchases_var.get()),
            'level_reached': int(self.level_var.get())
        }

        df_in = pd.DataFrame([prof])
        pred_class = self.pipeline.predict(df_in)[0]
        pred_probs = self.pipeline.predict_proba(df_in)[0]

        ret_pct = pred_probs[1] * 100.0
        churn_pct = pred_probs[0] * 100.0

        self.prob_bar.set(pred_probs[1])

        if pred_class == 1:
            self.card_result.configure(border_color=COLOR_EMERALD)
            self.res_badge.configure(text="STATUS: HIGH CONFIDENCE RETENTION", text_color=COLOR_EMERALD_LIGHT)
            self.res_outcome.configure(text="🛡️ RETAINED (WILL RETURN)", text_color="#ffffff")
            self.prob_bar.configure(progress_color=COLOR_EMERALD)
        else:
            self.card_result.configure(border_color=COLOR_GOLD)
            self.res_badge.configure(text="ALERT: CRITICAL RETENTION RISK", text_color=COLOR_GOLD_LIGHT)
            self.res_outcome.configure(text="⚠️ AT-RISK (WILL CHURN)", text_color=COLOR_GOLD_LIGHT)
            self.prob_bar.configure(progress_color="#f59e0b")

        self.res_prob_lbl.configure(text=f"Retention Probability: {ret_pct:.1f}%   |   Churn Risk: {churn_pct:.1f}%")

        # Live-Ops advice updates
        advices = []
        if prof['days_since_last_login'] > 5:
            advices.append("• 🚨 Inactivity Danger (Day 5+ Cliff): Inactive for >5 days. Dispatch automated push notification with 48h reward crate.")
        else:
            advices.append("• ✅ Recency Healthy: Recent login detected within 48h window. Maintain standard events.")

        if prof['days_active_last_30'] < 8:
            advices.append("• 📅 Habit Deficit: Active on <8 days this month. Offer consecutive daily login streak incentives.")

        if prof['in_game_purchases'] == 0:
            advices.append("• 💎 Monetization Gap: Zero microtransactions. Serve accessible $0.99 Starter Pack to establish commitment anchor.")
        else:
            advices.append("• 🏆 Invested Player: Microtransactions completed. Provide VIP battle pass tier skip incentives.")

        if prof['level_reached'] < 12:
            advices.append("• 🚀 FTUE Progression Friction: Low level. Check for difficulty spikes in chapters 2–3 and grant double-XP weekend pass.")

        advices.append("\n⚠️ Note: Predictions indicate statistical correlations. Always validate with controlled A/B testing.")
        self.advice_text.configure(text="\n".join(advices))

        # Update Radar Chart
        self.update_radar(prof)

    def update_radar(self, prof):
        self.ax.clear()
        self.ax.set_facecolor(COLOR_CARD)

        cats = ['Playtime', 'Sessions', 'Duration', 'Active', 'Level', 'Purchases']
        p_vals = [
            (prof['playtime_hours'] / self.max_vals['playtime_hours']) * 100,
            (prof['sessions_per_week'] / self.max_vals['sessions_per_week']) * 100,
            (prof['avg_session_duration'] / self.max_vals['avg_session_duration']) * 100,
            (prof['days_active_last_30'] / self.max_vals['days_active_last_30']) * 100,
            (prof['level_reached'] / self.max_vals['level_reached']) * 100,
            (min(prof['in_game_purchases'], 10) / 10.0) * 100
        ]
        c_vals = [
            (self.retained_means['playtime_hours'] / self.max_vals['playtime_hours']) * 100,
            (self.retained_means['sessions_per_week'] / self.max_vals['sessions_per_week']) * 100,
            (self.retained_means['avg_session_duration'] / self.max_vals['avg_session_duration']) * 100,
            (self.retained_means['days_active_last_30'] / self.max_vals['days_active_last_30']) * 100,
            (self.retained_means['level_reached'] / self.max_vals['level_reached']) * 100,
            (min(self.retained_means['in_game_purchases'], 10) / 10.0) * 100
        ]

        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        p_vals_c = p_vals + [p_vals[0]]
        c_vals_c = c_vals + [c_vals[0]]
        angles_c = angles + [angles[0]]

        self.ax.plot(angles_c, p_vals_c, color=COLOR_GOLD_LIGHT, linewidth=2.5, label='Current Player')
        self.ax.fill(angles_c, p_vals_c, color=COLOR_GOLD, alpha=0.3)

        self.ax.plot(angles_c, c_vals_c, color=COLOR_EMERALD, linewidth=1.8, linestyle='dashed', label='Retained Benchmark')
        self.ax.fill(angles_c, c_vals_c, color=COLOR_EMERALD, alpha=0.15)

        self.ax.set_xticks(angles)
        self.ax.set_xticklabels(cats, color='#ffffff', size=9, weight='bold')
        self.ax.set_yticks([25, 50, 75, 100])
        self.ax.set_yticklabels(['', '', '', ''], color='#64748b', size=7)
        self.ax.grid(color=(1.0, 1.0, 1.0, 0.15), linewidth=0.8)
        self.ax.spines['polar'].set_color(COLOR_EMERALD)
        self.ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, facecolor=COLOR_CARD, edgecolor='none', labelcolor='#ffffff', fontsize=8)

        self.canvas.draw()

if __name__ == "__main__":
    app = ApexRetentionApp()
    app.mainloop()
