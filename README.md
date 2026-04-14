# Financial News Sentiment Analyzer — FinBERT + Stock Price Signals

An NLP pipeline that scores financial news headlines using FinBERT, aggregates daily sentiment per company, maps it to next-day stock price returns, and measures signal quality using the Information Coefficient (IC). Deployed as a multi-page Streamlit dashboard with heatmaps, lag correlation analysis, and a backtested long/short strategy.

---

## What This Project Does

Sentiment analysis on financial news is one of the most widely used alternative data signals in quantitative equity research. This project builds the complete pipeline:

- **News Collection:** NewsAPI or RSS feeds (MoneyControl, ET Markets, Business Standard) — no paid data required
- **FinBERT Scoring:** Domain-specific BERT model fine-tuned on financial text. Outputs Positive / Neutral / Negative + confidence score per headline
- **Signal Aggregation:** Three aggregation methods — simple mean, confidence-weighted mean (primary), and polarity ratio
- **IC Analysis:** Spearman rank correlation between sentiment signal and forward returns across multiple lag periods (0–7 days)
- **Backtest:** Naive long/short strategy — long high-sentiment stocks, short low-sentiment stocks — with cumulative return, Sharpe ratio, and win rate
- **Streamlit Dashboard:** Sentiment heatmap across companies and dates, sentiment vs price overlay chart, IC comparison, alert queue of recent scored headlines

---

## Why FinBERT Over VADER

General-purpose sentiment models fail on financial text because financial language is domain-specific:

| Phrase | VADER | FinBERT |
|--------|-------|---------|
| "margin compression" | Neutral | Negative |
| "order book strength" | Neutral | Positive |
| "below guidance" | Neutral | Negative |
| "beat estimates" | Neutral | Positive |
| "impairment charge" | Neutral | Negative |

FinBERT was fine-tuned on the Financial PhraseBank corpus and Reuters financial news, giving it the vocabulary to understand analyst language, earnings commentary, and corporate action descriptions correctly.

---

## Project Structure

```
sentiment_analyzer/
├── news_collector.py      # NewsAPI and RSS feed collectors
├── sentiment_engine.py    # FinBERT scoring pipeline
├── signal_builder.py      # Merge sentiment with price data, compute IC
├── main.py                # Full pipeline runner
├── dashboard.py           # Multi-page Streamlit dashboard
├── config.py              # API keys and company watchlist
├── data/
│   ├── headlines.csv      # Raw headlines with metadata
│   ├── sentiment.csv      # FinBERT-scored headlines
│   ├── signals.csv        # Daily sentiment merged with returns
│   ├── ic_results.csv     # IC by company and overall
│   └── backtest.csv       # Daily strategy returns
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-price-signals.git
cd sentiment-price-signals
```

### 2. Install dependencies

```bash
pip install transformers torch newsapi-python yfinance pandas numpy \
            scipy plotly streamlit scikit-learn requests feedparser
```

### 3. Get a NewsAPI key (free)

Sign up at [newsapi.org](https://newsapi.org) — takes 2 minutes. Free tier gives 100 requests/day and 1 month of historical data.

Add your key to `config.py`:

```python
NEWSAPI_KEY = "your_key_here"
```

### 4. Run the pipeline

```bash
python main.py
```

Downloads FinBERT (~440MB, one time), fetches headlines, scores all articles, merges with price data, computes IC, runs backtest. Saves all outputs to `data/`.

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`.

---

## Data Pipeline

```
NewsAPI / RSS Feeds
        ↓
Company Tagging (headline → ticker mapping)
        ↓
FinBERT Sentiment Scoring
(Positive / Neutral / Negative + confidence 0–1)
        ↓
Daily Aggregation per Company
  ├── Simple mean
  ├── Confidence-weighted mean (primary signal)
  └── Polarity ratio: (pos − neg) / total
        ↓
yfinance Price Data (NSE tickers + .NS suffix)
        ↓
Signal Dataset
(sentiment_t merged with return_t+1)
        ↓
IC Analysis + Backtest
```

---

## Companies Tracked (Default Watchlist)

| Company | Ticker | Sector |
|---------|--------|--------|
| Reliance Industries | RELIANCE.NS | Conglomerate |
| Tata Consultancy Services | TCS.NS | IT Services |
| Infosys | INFY.NS | IT Services |
| HDFC Bank | HDFCBANK.NS | Banking |
| Wipro | WIPRO.NS | IT Services |
| ICICI Bank | ICICIBANK.NS | Banking |
| Bajaj Finance | BAJFINANCE.NS | NBFC |
| Asian Paints | ASIANPAINT.NS | Consumer |
| Maruti Suzuki | MARUTI.NS | Auto |
| Sun Pharma | SUNPHARMA.NS | Pharma |

Easily extended — add any company name + NSE ticker to `WATCHLIST` in `config.py`.

---

## Signal Aggregation

Three approaches are computed and available in the dashboard:

**1. Confidence-Weighted Mean (primary)**
Each article's sentiment is weighted by FinBERT's confidence score. More certain predictions contribute more. This is the most reliable signal because FinBERT's accuracy is highest on high-confidence outputs.

```python
sentiment_wavg = np.average(numeric_scores, weights=confidence_scores)
```

**2. Simple Mean**
Equal weight to all articles. Simple but noisy — a flood of neutral news stories dilutes strong signals.

**3. Polarity Ratio**
```
polarity = (positive_articles − negative_articles) / total_articles
```
Ranges from −1 (all negative) to +1 (all positive). Useful for comparing sentiment direction across companies with different article volumes.

---

## Information Coefficient (IC)

IC is the standard quant finance metric for signal quality. It measures Spearman rank correlation between the signal and the forward return:

```
IC = Spearman(sentiment_t, return_t+1)
```

| IC Value | Interpretation |
|----------|---------------|
| > 0.10 | Strong signal — rare in practice |
| 0.05–0.10 | Useful signal — worth including in a model |
| 0.02–0.05 | Weak but meaningful signal |
| < 0.02 | Noise — not statistically distinguishable from zero |

IC is computed at multiple lag periods (0–7 days) to show the signal decay curve — how quickly news is incorporated into prices.

---

## Backtest Strategy

A naive long/short strategy to validate the signal:

- **Long:** Companies with daily sentiment > +threshold
- **Short:** Companies with daily sentiment < −threshold
- **Hold:** 1 day
- **Weighting:** Equal weight across all signals on each day

**Important caveats:**
- No transaction costs, no market impact, no slippage
- This is signal validation, not a tradable strategy
- A real implementation requires execution infrastructure, risk limits, and regulatory approval

**Reported metrics:** Cumulative return, Sharpe ratio (annualised), win rate, average signals per day.

---

## Dashboard Pages

| Page | What It Shows |
|------|--------------|
| Overview | Total articles, IC, strategy return summary metrics |
| Company Deep Dive | Sentiment bars + price line overlay, IC for selected company, scored headlines table |
| IC Analysis | Bar chart of IC by company, lag correlation decay curve |
| Sentiment Heatmap | All companies × all dates — colour gradient from red (negative) to green (positive) |
| Backtest | Cumulative return chart, daily P&L, Sharpe and win rate metrics |
| Recent Headlines | Latest scored headlines with FinBERT label colour-coding |

---

## Free Data Alternatives (If NewsAPI Quota Runs Out)

RSS feeds require no API key and have no daily limit:

```python
RSS_FEEDS = {
    "Economic Times":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol":    "https://www.moneycontrol.com/rss/latestnews.xml",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "Mint Markets":    "https://www.livemint.com/rss/markets",
}
```

Install feedparser: `pip install feedparser`

---

## Limitations

- NewsAPI free tier limited to 100 requests/day and 1 month history — limits dataset size
- FinBERT was trained primarily on English financial text from US/UK sources — may have reduced accuracy on Indian-specific terminology and company names
- Signal IC is measured on a small sample (10 companies, 28 days) — production systems use hundreds of companies and years of history
- Backtest does not account for transaction costs, execution delay, or bid-ask spread
- Headline-level scoring — full article text would improve accuracy but requires paid NewsAPI plan or web scraping

---

## Potential Extensions

- [ ] Full article text scoring (not just headlines) for better accuracy
- [ ] Earnings call transcript sentiment (conference call NLP)
- [ ] Multi-language support for Hindi financial news
- [ ] Real-time scoring pipeline with Kafka or Celery
- [ ] Combine with price momentum factor for multi-factor signal
- [ ] Sector-level aggregation — is IT sector sentiment bullish vs Banking?
- [ ] Event-driven analysis: earnings announcements, RBI policy days

---

## Tech Stack

| Category | Library |
|----------|---------|
| NLP | transformers (HuggingFace), torch |
| Model | ProsusAI/finbert |
| Data | newsapi-python, feedparser, yfinance |
| Processing | pandas, numpy |
| Statistics | scipy (Spearman correlation) |
| Visualization | plotly |
| Dashboard | streamlit |

---

## References

- Malo, P. et al. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.* Journal of the American Society for Information Science and Technology. — Financial PhraseBank dataset used to train FinBERT
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063. — Original FinBERT paper
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. — Base BERT architecture
- Grinold, R. & Kahn, R. (1994). *Active Portfolio Management.* — Information Coefficient and signal quality framework
- NSE India: https://nseindia.com — NSE ticker data
- NewsAPI: https://newsapi.org — News headline API

---

## Author

**Deeraj**
PGDM-BFS, IMT Ghaziabad (Batch 2025–27)
B.Tech Mechanical Engineering, NIT Calicut
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
