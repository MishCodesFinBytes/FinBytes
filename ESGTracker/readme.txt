# ESG Score Tracking (Python CLI Demo)

A simple Python CLI tool demonstrating **ESG (Environmental, Social, Governance) score tracking** using historical data.

This project is based on a fintech blog post and demonstrates:
- ESG data ingestion
- Score calculation
- Historical tracking by date
- CLI-based interaction
- Automated testing

---

## 🚀 Features

- Query ESG score by company and date
- Weighted ESG scoring model
- Simple and reproducible dataset
- Pytest-based test suite

---

## 📂 Project Structure
esg-tracker/
│
├── esg_tracker_demo.py # CLI application
├── esg_tracker_caller.py # Test cases
└── README.md


---

## ▶️ Usage

### Run the CLI

```bash
python demo.py --company AAPL --date 2023-01
Example Output
Company: AAPL
Date: 2023-01
E: 78 S: 72 G: 80
ESG Score: 77.6
🧪 Run Tests

Install dependencies:

pip install -r pytest

Run tests:

pytest -v
🧠 ESG Scoring Logic

The ESG score is calculated as:

Environmental (E): 40%
Social (S): 30%
Governance (G): 30%
