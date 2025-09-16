# Indian Market Bot (SmartAPI + GPT)

## Features
- Multi-timeframe analysis (5m / 15m / 30m)
- Price action signals (Entry, SL, Targets)
- GPT-4o-mini candlestick & chart pattern annotations
- Telegram alerts with charts
- Manual start/stop (optimized for Railway)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file (see `.env.example`) with your SmartAPI, Telegram, and OpenAI keys.

3. Run locally:
   ```bash
   python main.py
   ```

4. On Railway, deploy and use `Procfile`:
   - **Start** worker at 8:50 AM (pre-open).
   - **Stop** worker at 3:30 PM (after market close).

## Files
- `main.py` : Bot logic
- `requirements.txt` : Dependencies
- `.env.example` : Example config
- `Procfile` : Railway worker definition
- `README.md` : Instructions
