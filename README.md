# AI Traffic Demand Predictor

> End-to-end ML system for cyclical traffic demand forecasting
> Built for Flipkart Gridlock Hackathon 2.0 | Extended to Production

## 🏆 Results
- Leaderboard Score: 91+ | Top 15% (~1200/8000+ teams)
- OOF R² Score: 0.955 (consistent 0.951–0.957 across 5 folds)

## 🔗 Links
- Live Demo: [traffic-demand-frontend.vercel.app](https://traffic-demand-frontend.vercel.app)
- Dataset: Flipkart Gridlock Hackathon 2.0

## 🛠️ Tech Stack
- ML: LightGBM, Scikit-Learn, Pandas, NumPy
- API: FastAPI, Python
- Frontend: React, Vercel
- Deployment: Render (backend), Vercel (frontend)

## 📊 Model Pipeline
1. Feature Engineering (cyclical encoding, OOF target encoding)
2. Infrastructure & weather feature integration
3. LightGBM with 5-fold cross-validation
4. FastAPI inference endpoint
5. OOD exception handling

## 🚀 Local Setup
```bash
git clone https://github.com/Sujeet12000/traffic-demand-prediction
cd traffic-demand-prediction
pip install -r requirements.txt
python api.py
```

## 📸 Screenshots
[Add 2-3 screenshots here]
