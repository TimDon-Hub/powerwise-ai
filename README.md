# PowerWise AI
### Smart Home Energy Management System

## 📌 Problem Statement

In Nigeria and across Africa, households face three critical electricity challenges:

- **Unstable supply** — frequent outages from national grid providers
- **High cost** — prepaid meter systems that drain quickly
- **Poor management** — appliances left on when not in use, no visibility into consumption

These lead to unnecessary energy waste, higher bills, and difficulty planning around limited power availability.

---

## 💡 Our Solution — PowerWise AI

PowerWise AI is a **smart home energy management system** that uses **simulation + machine learning** to help Nigerian households:

- Monitor and control household devices in real time
- Automatically shut off idle devices (AI-driven)
- Visualise energy consumption per device and over time
- Predict how long prepaid units will last using ML
- Receive personalised energy-saving recommendations

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| Backend Logic | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Linear Regression) |
| Visualisation | Plotly |
| Version Control | GitHub |

---

## 🤖 ML Approach

We train a **Linear Regression** model on 30 days of synthetic (simulated) household consumption data.

**Features used:**
- `day_index` — captures consumption trend over time
- `is_weekend` — weekend vs weekday usage pattern
- `rolling_3day` — 3-day rolling average (momentum)

**Outputs:**
- Predicted daily consumption for the next 30 days
- Estimated number of days until prepaid units run out
- Projected recharge date

---

## 🗂️ Project Structure

```
powerwise_ai/
├── app.py            # Main Streamlit application (4 tabs)
├── devices.py        # Device simulation layer (Device class)
├── tracker.py        # Energy tracking & historical data generation
├── ml_model.py       # ML training, predictions & recommendations
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/powerwise-ai.git
cd powerwise-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📱 App Features

### 🏠 Dashboard Tab
- KPI cards: units remaining, cost so far, active devices, efficiency score
- Real-time smart alerts (idle device shutoffs)
- Quick device overview table
- Low-unit warnings

### 🔌 Device Control Tab
- Toggle individual devices ON/OFF grouped by room
- "I'm Here" button to reset idle timer and prevent auto-shutoff
- Session event log
- Bulk ON/OFF controls

### 📊 Analytics Tab
- Bar chart: energy consumption per device
- Pie chart: cost share per device
- 30-day historical consumption area chart
- Weekly usage pattern (Mon–Sun)
- Full device report table with time-on, kWh, and cost

### 🤖 AI Predictions Tab
- Model performance metrics (MAE, R²)
- Predicted days until units run out
- Historical vs AI-predicted consumption chart
- Unit depletion curve with recharge date marker
- Personalised AI recommendations

---

## 🏠 Simulated Devices

| Device | Room | Wattage |
|--------|------|---------|
| Living Room Light | Living Room | 60W |
| Bedroom Light | Bedroom | 40W |
| Kitchen Light | Kitchen | 40W |
| Living Room Fan | Living Room | 75W |
| Bedroom Fan | Bedroom | 75W |
| Television | Living Room | 120W |
| Electric Iron | Utility | 1000W |
| Phone Charger | Bedroom | 15W |

---

## 👥 Team

Built by a team of Lagos State University engineering students:
- (Timothy Donald)[https://linkedin.com/in/timdon369]
- (Obedience Adara)[https://linkedin.com/in/obedienceadara]
- (Ejikeme Victor)[https://linkedin.com/in/ejikemevictor]
- Omgbrumaye Praise

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*PowerWise AI — helping Nigerian homes use less, save more, and live smarter. ⚡*
