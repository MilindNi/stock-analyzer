# @title

# !pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response
from plotly.subplots import make_subplots
import os
from fastapi.responses import FileResponse

from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your Lovable URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import warnings
import plotly.graph_objs as go
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
import pytz

warnings.filterwarnings('ignore')

class ForexAnalyzer:
    def __init__(self, username=None, password=None):
        if username and password:
            self.tv = TvDatafeed(username, password)
        else:
            self.tv = TvDatafeed()
        self.data = None

    def get_data(self, symbol, exchange='FX', interval=Interval.in_daily, max_bars=15000, fetch_bars=15000):
        all_data = pd.DataFrame()
        # fetch_bars = 500  # max per request
        last_timestamp = None
        eastern = pytz.timezone('America/New_York')
        # start_date = eastern.localize(datetime(2025, 10, 28)).astimezone(pytz.UTC)

        while len(all_data) < max_bars:
            bars_to_fetch = min(fetch_bars, max_bars - len(all_data))

            df = self.tv.get_hist(symbol.upper(), exchange.upper(), interval, bars_to_fetch)
            if df.empty:
                break

            # Standardize column casing
            df.columns = df.columns.str.title()

            # eastern = pytz.timezone('America/New_York')
            exchange = exchange.upper()
            # if exchange == 'FX':
            #   tz_exchange = 'UTC'
            # elif exchange in ['NASDAQ', 'NYSE']:
            #   tz_exchange = 'America/New_York'
            # elif exchange in ['NSE', 'BSE']:
            #   tz_exchange = 'Asia/Kolkata'
            # elif exchange == 'BINANCE':
            #   tz_exchange = 'UTC'
            # else:
            #   tz_exchange = 'UTC'

            # if tz_exchange in ['Asia/Kolkata']:
            #   df.index = pd.to_datetime(df.index) + pd.Timedelta(hours=5, minutes=30)
            # else:
            #   df.index = pd.to_datetime(df.index)
            # df.index = pd.to_datetime(df.index).tz_localize(tz_exchange)
            df.index = pd.to_datetime(df.index, utc=False)
            if exchange in ['NSE', 'BSE']:
              df.index = df.index + pd.Timedelta(hours=5, minutes=30)



            # df.index = df.index.tz_convert('UTC')  # Now safe to convert
            # df.index = df.index.tz_localize(eastern, ambiguous='NaT', nonexistent='shift_forward')
            # df.index = df.index.tz_convert('UTC')
            # if exchange.upper() in ['FX', 'BINANCE', 'OANDA']:  # UTC exchanges
            #   df.index = pd.to_datetime(df.index).tz_localize('UTC')
            # elif exchange.upper() in ['NASDAQ', 'NYSE']:  # US stocks
            #   df.index = pd.to_datetime(df.index).tz_localize('America/New_York').tz_convert('UTC')
            # elif exchange.upper() in ['NSE', 'BSE']:  # India
            #   df.index = pd.to_datetime(df.index).tz_localize('Asia/Kolkata').tz_convert('UTC')
            # else:
            #   df.index = pd.to_datetime(df.index).tz_localize('UTC')

            # Remove overlap if paging
            if last_timestamp:
                df = df[df.index < last_timestamp]
            if df.empty:
                break

            all_data = pd.concat([df, all_data]).sort_index()
            last_timestamp = all_data.index.min()

        all_data = all_data.tail(max_bars)
        self.data = all_data
        print(f"Fetched total {len(all_data)} data points for {symbol} in GMT/UTC time with {interval} interval.")
        # print(all_data.head(5))
        return all_data

    def calculate_indicators(self):
        c = self.data['Close']; h = self.data['High']; l = self.data['Low']; o = self.data['Open']
        self.data['SMA_20'] = c.rolling(window=20).mean()
        self.data['SMA_50'] = c.rolling(window=50).mean()
        self.data['EMA_12'] = c.ewm(span=12).mean()
        self.data['EMA_26'] = c.ewm(span=26).mean()
        macd_line = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD'] = macd_line
        self.data['MACD_signal'] = macd_line.ewm(span=9).mean()
        self.data['RSI'] = self._rsi(c, 14)
        sma20 = self.data['SMA_20']
        std = c.rolling(20).std()
        self.data['BB_upper'] = sma20 + 2*std
        self.data['BB_lower'] = sma20 - 2*std
        self.data['ADX'] = self._adx(h, l, c, 14)
        self.data['ATR'] = self._atr(h, l, c, 14)
        self._detect_sequential_hh_hl()
        self._detect_sequential_ll_lh()
        self._detect_break_of_prev_higher_low()
        self.compute_count_of_higher_low_breaches()
        self._detect_break_of_prev_lower_high()
        self.compute_count_of_lower_high_breaches()
        self._fill_prev_higher_low()
        self._detect_patterns()
        self.detect_candlestick_patterns()

    # def detect_gaps(self, pct_threshold=0.003):  # 0.1% threshold by default
    #     prev_high = self.data['Close'].shift(1)
    #     curr_open = self.data['Open']
    #     # if(curr_open > prev_high):
    #     gap_size = (curr_open - prev_high) / prev_high
    #     gaps = gap_size.abs() > pct_threshold
    #     self.data['Is_Gap'] = gaps.astype(int)
    #     self.data['Gap_Size'] = gap_size.where(gaps, 0)

    def detect_gaps(self, pct_threshold=0.001):
        d = self.data

        prev_high = d['High'].shift(1)
        prev_low = d['Low'].shift(1)
        curr_low = d['Low']
        curr_high = d['High']

        # Full gap-up: today's low above previous high
        raw_gap_up = curr_low - prev_high
        raw_gap_down = prev_low - curr_high

        gap_up_pct = raw_gap_up / prev_high
        gap_down_pct = raw_gap_down / prev_low

        is_gap_up = (curr_low > prev_high) & (gap_up_pct >= pct_threshold)
        is_gap_down = (curr_high < prev_low) & (gap_down_pct >= pct_threshold)

        # d['Gap_Pct'] = 0.0
        # d.loc[is_gap_up, 'Gap_Pct'] = gap_up_pct[is_gap_up]
        # d.loc[is_gap_down, 'Gap_Pct'] = -gap_down_pct[is_gap_down]

        d['Is_Gap_Up'] = is_gap_up.astype(int)
        d['Is_Gap_Down'] = is_gap_down.astype(int)
        d['Is_Gap'] = (is_gap_up | is_gap_down).astype(int)

        # # Gap zone strictly between prior and current ranges
        d['Gap_Zone_Low'] = np.where(is_gap_up, prev_high,
                              np.where(is_gap_down, curr_high, np.nan))
        d['Gap_Zone_High'] = np.where(is_gap_up, curr_low,
                              np.where(is_gap_down, prev_low, np.nan))


    def detect_explosive_gap_buys(self, pct_threshold=0.01, lookback_bars=100):
        d = self.data
        n = len(d)
        explosive = np.zeros(n, dtype=int)

        active_gap_low = None
        active_gap_high = None
        gap_active = False

        for i in range(n):
            # If current bar is a new gap-up, start tracking its zone
            if d['Is_Gap_Up'].iloc[i] == 1:
                active_gap_low = d['Gap_Zone_Low'].iloc[i]
                active_gap_high = d['Gap_Zone_High'].iloc[i]
                gap_active = True
                # do NOT mark explosive on the gap bar itself
                continue

            if not gap_active:
                continue

            bar_low = d['Low'].iloc[i]
            bar_high = d['High'].iloc[i]

            # "Hover/touch" condition: any overlap between candle range and gap zone
            if (bar_low <= active_gap_high) and (bar_high >= active_gap_low):
                explosive[i] = 1
                # deactivate after the FIRST touch
                gap_active = False
                active_gap_low = None
                active_gap_high = None

        d['Explosive_Gap_Buy'] = explosive
        self.data = d


    def _rsi(self, s, window):
        delta = s.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - 100 / (1 + rs)

    def _atr(self, h, l, c, window):
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _adx(self, h, l, c, window):
        tr = self._atr(h, l, c, window)
        plus_dm = np.where((h.diff() > l.diff().abs()) & (h.diff() > 0), h.diff(), 0)
        minus_dm = np.where((l.diff().abs() > h.diff()) & (l.diff() < 0), l.diff().abs(), 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(window).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(window).mean() / tr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        return dx.rolling(window).mean().fillna(0)

    def _detect_sequential_hh_hl(self):
        """
        Higher_High / Higher_Low detection (next‑bar confirmation).

        HH rule:
          - Previous confirmed HH has high H_prev (or -inf initially).
          - A bar i (0 <= i < n-1) is a new HH if:
              High[i] > H_prev  AND  Close[i+1] < Close[i]
          - The last bar (i = n-1) is ignored for confirmation.

        HL rule:
          - For each pair of consecutive HHs at indices h1 < h2,
            HL is the bar k with the LOWEST Low in (h1, h2).
        """
        # import numpy as np

        high  = self.data['High'].values
        low   = self.data['Low'].values
        close = self.data['Close'].values
        n = len(self.data)

        HH = np.zeros(n, dtype=int)
        HL = np.zeros(n, dtype=int)

        if n < 2:
            self.data['Higher_High'] = HH
            self.data['Higher_Low'] = HL
            return

        # 1) confirm HHs with next‑bar close rule
        last_hh_high = -np.inf
        hh_indices = []

        for i in range(n - 1):  # up to n-2, because we look at i+1
            if high[i] > last_hh_high and low[i+1] < low[i] and high[i+1] < high[i]:
                HH[i] = 1
                hh_indices.append(i)
                last_hh_high = high[i]

        # 2) HL = deepest low between consecutive HHs
        if len(hh_indices) >= 2:
            for h1, h2 in zip(hh_indices[:-1], hh_indices[1:]):
                start = h1 + 1
                end = h2
                if end <= start:
                    continue
                seg_lows = low[start:end]
                if seg_lows.size == 0:
                    continue
                deep_rel = int(np.argmin(seg_lows))
                deep_idx = start + deep_rel
                if 0 <= deep_idx < n:
                    HL[deep_idx] = 1

        self.data['Higher_High'] = HH
        self.data['Higher_Low'] = HL














    # def _detect_sequential_hh_hl(self):
    #     h = self.data['High']
    #     l = self.data['Low']
    #     o = self.data['Open']
    #     c = self.data['Close']

    #     n = len(self.data)
    #     hh = [0] * n
    #     hl = [0] * n

    #     last_confirmed_hh_idx = None      # index of last confirmed HH
    #     last_confirmed_hh_value = -np.inf

    #     pending_hh_idx = None             # index of highest bar since last confirmed HH
    #     pending_hh_value = -np.inf

    #     # for HL between two confirmed HHs
    #     trough_low = np.inf
    #     trough_idx = None

    #     for i in range(n):
    #         # 1) Update pending HH (break of previous confirmed HH)
    #         if h.iloc[i] > last_confirmed_hh_value:
    #             # if no pending HH yet or this bar makes a higher high than current pending HH
    #             if pending_hh_idx is None or h.iloc[i] > pending_hh_value:
    #                 pending_hh_idx = i
    #                 pending_hh_value = h.iloc[i]

    #         # 2) While we have at least one confirmed HH and no new one confirmed yet,
    #         #    track the deepest low after the last confirmed HH
    #         if last_confirmed_hh_idx is not None:
    #             if trough_low == np.inf:
    #                 # start tracking after the last confirmed HH
    #                 if i > last_confirmed_hh_idx:
    #                     trough_low = l.iloc[i]
    #                     trough_idx = i
    #             else:
    #                 if i > last_confirmed_hh_idx and l.iloc[i] < trough_low:
    #                     trough_low = l.iloc[i]
    #                     trough_idx = i

    #         # 3) Confirm pending HH on a red candle AFTER the pending bar
    #         if pending_hh_idx is not None:
    #             is_red = c.iloc[i] < o.iloc[i]
    #             if is_red and i > pending_hh_idx:
    #                 # confirm HH at the pending bar
    #                 hh[pending_hh_idx] = 1

    #                 # if there was a previous HH, mark HL as deepest low between the two HHs
    #                 if last_confirmed_hh_idx is not None and trough_idx is not None:
    #                     if last_confirmed_hh_idx < trough_idx < pending_hh_idx:
    #                         hl[trough_idx] = 1

    #                 # update last confirmed HH
    #                 last_confirmed_hh_idx = pending_hh_idx
    #                 last_confirmed_hh_value = pending_hh_value

    #                 # reset for next leg
    #                 pending_hh_idx = None
    #                 pending_hh_value = -np.inf
    #                 trough_low = np.inf
    #                 trough_idx = None

    #     self.data['Higher_High'] = hh
    #     self.data['Higher_Low'] = hl




    def _detect_sequential_ll_lh(self):
        """
        Lower_Low / Lower_High detection (next‑bar confirmation).

        LL rule:
          - Previous confirmed LL has low L_prev (or +inf initially).
          - A bar i (0 <= i < n-1) is a new LL if:
              Low[i] < L_prev  AND  Close[i+1] > Close[i]
          - The last bar (i = n-1) is ignored for confirmation.

        LH rule:
          - For each pair of consecutive LLs at indices l1 < l2,
            LH is the bar k with the HIGHEST High in (l1, l2).
        """
        # import numpy as np

        high  = self.data['High'].values
        low   = self.data['Low'].values
        close = self.data['Close'].values
        n = len(self.data)

        LL = np.zeros(n, dtype=int)
        LH = np.zeros(n, dtype=int)

        if n < 2:
            self.data['Lower_Low'] = LL
            self.data['Lower_High'] = LH
            return

        # 1) confirm LLs with next‑bar close rule
        last_ll_low = np.inf
        ll_indices = []

        for i in range(n - 1):  # up to n-2, because we look at i+1
            if low[i] < last_ll_low and close[i+1] > close[i] and high[i+1] > high[i]:
                LL[i] = 1
                ll_indices.append(i)
                last_ll_low = low[i]

        # 2) LH = highest high between consecutive LLs
        if len(ll_indices) >= 2:
            for l1, l2 in zip(ll_indices[:-1], ll_indices[1:]):
                start = l1 + 1
                end = l2
                if end <= start:
                    continue
                seg_highs = high[start:end]
                if seg_highs.size == 0:
                    continue
                peak_rel = int(np.argmax(seg_highs))
                peak_idx = start + peak_rel
                if 0 <= peak_idx < n:
                    LH[peak_idx] = 1

        self.data['Lower_Low'] = LL
        self.data['Lower_High'] = LH





    def _detect_break_of_prev_higher_low(self):
        hl_indices = self.data[self.data['Higher_Low'] == 1].index
        last_hl = None
        breaks = [np.nan] * len(self.data)
        broken = False
        for idx in self.data.index:
            loc = self.data.index.get_loc(idx)
            if self.data.at[idx, 'Higher_Low'] == 1:
                last_hl = idx
                broken = False
            if last_hl and idx != last_hl and not broken:
                # atr = self.data.at[idx, 'ATR'] if pd.notna(self.data.at[idx, 'ATR']) else 0
                break_threshold = self.data.at[last_hl, 'Low']
                if self.data.at[idx, 'Low'] < break_threshold:
                    breaks[loc] = 1
                    broken = True
                else:
                    breaks[loc] = 0
            else:
                breaks[loc] = np.nan
        self.data['Break_Prev_Higher_Low'] = breaks

    def _detect_break_of_prev_lower_high(self):
    # import numpy as np
    # import pandas as pd

        lh_indices = self.data[self.data['Lower_High'] == 1].index
        last_lh = None
        breaks = [np.nan] * len(self.data)
        broken = False

        for idx in self.data.index:
            loc = self.data.index.get_loc(idx)

            # when a new LH appears, reset reference and broken flag
            if self.data.at[idx, 'Lower_High'] == 1:
                last_lh = idx
                broken = False

            # check for breach of last LH's high
            if last_lh is not None and idx != last_lh and not broken:
                break_threshold = self.data.at[last_lh, 'High']
                if self.data.at[idx, 'High'] > break_threshold:
                    breaks[loc] = 1
                    broken = True
                else:
                    breaks[loc] = 0
            else:
                breaks[loc] = np.nan

        self.data['Break_Prev_Lower_High'] = breaks


    def compute_count_of_higher_low_breaches(self):
        hl_indices = self.data[self.data['Higher_Low'] == 1].index.tolist()
        count_breaches = [0] * len(self.data)
        for i in range(len(self.data)):
            close_price = self.data['Low'].iloc[i]
            breaches = 0
            # Only look back at higher lows that occurred before current point
            breached = False
            for k, hl_idx in enumerate(reversed(hl_indices)):
                hl_loc = self.data.index.get_loc(hl_idx)
                if hl_loc >= i:
                    continue  # Only check higher lows before this bar
                hl_value = self.data['Low'].iloc[hl_loc]
                # atr = self.data['ATR'].iloc[i] if 'ATR' in self.data.columns else 0
                break_threshold = hl_value
                if close_price < break_threshold:
                    breaches += 1
                    breached = True
                else:
                    break  # As soon as a non-breached one is found, stop
            count_breaches[i] = breaches
        self.data['Count_Higher_Low_Breaches'] = count_breaches

    def compute_count_of_lower_high_breaches(self):
        lh_indices = self.data[self.data['Lower_High'] == 1].index.tolist()
        count_breaches = [0] * len(self.data)

        for i in range(len(self.data)):
            high_price = self.data['High'].iloc[i]
            breaches = 0

            # Only look back at lower highs that occurred before current point
            for lh_idx in reversed(lh_indices):
                lh_loc = self.data.index.get_loc(lh_idx)
                if lh_loc >= i:
                    continue  # Only check lower highs before this bar

                lh_value = self.data['High'].iloc[lh_loc]
                break_threshold = lh_value

                if high_price > break_threshold:
                    breaches += 1
                else:
                    # As soon as we hit a non-breached LH, stop,
                    # since earlier LHs are even higher (in a proper downtrend)
                    break

            count_breaches[i] = breaches

        self.data['Count_Lower_High_Breaches'] = count_breaches


    def _fill_prev_higher_low(self):
        last_hl_idx = None
        prev_hl_list = []
        for idx in self.data.index:
            if self.data.at[idx, 'Higher_Low'] == 1:
                last_hl_idx = idx
            if last_hl_idx:
                prev_hl_list.append(self.data.at[last_hl_idx,'Low'])
            else:
                prev_hl_list.append(np.nan)
        self.data['Prev_Higher_Low'] = prev_hl_list

    def compute_primary_trend(self):
        """
        Primary trend from price slope (Close):
          - rolling regression slope over 'window' bars
          - UP if slope>0, DOWN if slope<0, SIDE if no slope yet
          Writes: PrimaryTrend per bar.
        """
        window = 30

        df = self.data.copy()
        if df is None or df.empty or "Close" not in df.columns:
            return

        close = df["Close"].values
        n = len(close)
        slopes = np.full(n, np.nan)

        x = np.arange(window)
        xm = x.mean()
        denom = ((x - xm) ** 2).sum()

        for i in range(window - 1, n):
            y = close[i - window + 1 : i + 1]
            ym = y.mean()
            num = ((x - xm) * (y - ym)).sum()
            slopes[i] = num / denom if denom != 0 else np.nan

        labels = []
        for s in slopes:
            if np.isnan(s):
                labels.append("SIDE")
            elif s > 0:
                labels.append("UP")
            elif s < 0:
                labels.append("DOWN")
            else:
                labels.append("SIDE")

        df["PrimaryTrend"] = labels
        self.data = df



    def summarize_primary_trend(self):
        """
        Uses PrimaryTrend to describe major legs:
          - General bias.
          - Major UP/DOWN legs and the last one.
        """
        min_len = 15

        df = self.data
        if df is None or df.empty or "PrimaryTrend" not in df.columns:
            return "Primary trend not computed."

        tr = df["PrimaryTrend"]

        # segment
        change_mask = tr != tr.shift()
        grp = change_mask.cumsum()
        segs = df.groupby(grp).agg(
            label=("PrimaryTrend", "first"),
            start=("PrimaryTrend", lambda s: s.index[0]),
            end=("PrimaryTrend",   lambda s: s.index[-1]),
            length=("PrimaryTrend", "size"),
        ).reset_index(drop=True)

        major = segs[
            (segs["label"].isin(["UP", "DOWN"])) & (segs["length"] >= min_len)
        ].reset_index(drop=True)

        if major.empty:
            return "No major primary trend segments detected."

        up = (tr == "UP").sum()
        down = (tr == "DOWN").sum()
        if up > down:
            general = "General uptrend over the period."
        elif down > up:
            general = "General downtrend over the period."
        else:
            general = "No clear dominant trend over the period."

        lines = [general]

        prev = None
        for _, row in major.iterrows():
            lab = row["label"]
            s = row["start"]
            e = row["end"]
            if prev in ("UP", "DOWN") and lab in ("UP", "DOWN") and lab != prev:
                lines.append(
                    f"Trend change {prev}->{lab} around {s.strftime('%d %b %Y %H:%M')}."
                )
            prev = lab

        last_ts = df.index[-1]
        last_trend = major.iloc[-1]["label"]
        last_start = major.iloc[-1]["start"]

        lines.append(
            f"Latest major trend is {last_trend} since {last_start.strftime('%d %b %Y %H:%M')} "
            f"(as of {last_ts.strftime('%d %b %Y %H:%M')})."
        )

        return " ".join(lines)




    def summarize_breaches_in_primary_legs(self):
        """
        For each major primary trend leg:
          - If UP: report first Higher_Low breach (Count_Higher_Low_Breaches >= 1).
          - If DOWN: report first Lower_High breach (Count_Lower_High_Breaches >= 1).
        """
        min_len = 15

        df = self.data
        needed = [
            "PrimaryTrend",
            "Count_Higher_Low_Breaches",
            "Count_Lower_High_Breaches",
        ]
        if df is None or df.empty or any(c not in df.columns for c in needed):
            return ""

        tr = df["PrimaryTrend"]

        # major legs as before
        change_mask = tr != tr.shift()
        grp = change_mask.cumsum()
        segs = df.groupby(grp).agg(
            label=("PrimaryTrend", "first"),
            start=("PrimaryTrend", lambda s: s.index[0]),
            end=("PrimaryTrend",   lambda s: s.index[-1]),
            length=("PrimaryTrend", "size"),
        ).reset_index(drop=True)

        major = segs[
            (segs["label"].isin(["UP", "DOWN"])) & (segs["length"] >= min_len)
        ].reset_index(drop=True)

        if major.empty:
            return ""

        lines = []

        for _, row in major.iterrows():
            lab = row["label"]
            s = row["start"]
            e = row["end"]
            leg_df = df.loc[s:e]

            if lab == "UP":
                # first HL breach in this UP leg
                breaches = leg_df[leg_df["Count_Higher_Low_Breaches"] >= 1]
                if not breaches.empty:
                    ts = breaches.index[0]
                    lines.append(
                        f"In UP leg from {s.strftime('%d %b %Y')} to {e.strftime('%d %b %Y')}, "
                        f"first HL breach on {ts.strftime('%d %b %Y %H:%M')}."
                    )
            elif lab == "DOWN":
                # first LH breach in this DOWN leg
                breaches = leg_df[leg_df["Count_Lower_High_Breaches"] >= 1]
                if not breaches.empty:
                    ts = breaches.index[0]
                    lines.append(
                        f"In DOWN leg from {s.strftime('%d %b %Y')} to {e.strftime('%d %b %Y')}, "
                        f"first LH breach on {ts.strftime('%d %b %Y %H:%M')}."
                    )

        return " ".join(lines)

    def detect_bullish_choch(self):
        df = self.data
        max_bars_after_lh = 40
        needed = {"High", "Low", "Close", "Lower_High", "Lower_Low"}
        if df is None or df.empty or not needed.issubset(df.columns):
            return pd.DataFrame()

        high    = df["High"]
        low     = df["Low"]
        close   = df["Close"]
        lh_flag = df["Lower_High"].fillna(0).astype(int)
        ll_flag = df["Lower_Low"].fillna(0).astype(int)

        idx        = df.index
        ll_indices = list(idx[ll_flag == 1])
        lh_indices = list(idx[lh_flag == 1])
        pos        = {t: i for i, t in enumerate(idx)}
        events     = []

        for lh_time in lh_indices:
            i_lh = pos[lh_time]

            # at least 1 LL before this LH = downtrend context
            prev_lls = [t for t in ll_indices if t < lh_time]
            if len(prev_lls) < 1:
                continue

            lh_price = high.loc[lh_time]

            # first close above LH
            choch_time = None
            for j in range(i_lh + 1, min(i_lh + 1 + max_bars_after_lh, len(df))):
                if close.iloc[j] > lh_price:
                    choch_time = idx[j]
                    break
            if choch_time is None:
                continue

            choch_close = close.loc[choch_time]

            # all LHs broken by this close
            broken_lhs = [
                t for t in lh_indices
                if t < choch_time and high.loc[t] < choch_close
            ]
            latest_lh_broken = broken_lhs[-1] if broken_lhs else lh_time

            events.append({
                "type"            : "Bullish_CHoCH",
                "choch_time"      : choch_time,
                "lh_time"         : lh_time,
                "latest_lh_broken": latest_lh_broken,
                "lhs_broken"      : len(broken_lhs),
            })

        return pd.DataFrame(events) if events else pd.DataFrame()


    def detect_bearish_choch(self):
        df = self.data
        max_bars_after_hl = 40
        needed = {"High", "Low", "Close", "Higher_High", "Higher_Low"}
        if df is None or df.empty or not needed.issubset(df.columns):
            return pd.DataFrame()

        high    = df["High"]
        low     = df["Low"]
        close   = df["Close"]
        hh_flag = df["Higher_High"].fillna(0).astype(int)
        hl_flag = df["Higher_Low"].fillna(0).astype(int)

        idx        = df.index
        hh_indices = list(idx[hh_flag == 1])
        hl_indices = list(idx[hl_flag == 1])
        pos        = {t: i for i, t in enumerate(idx)}
        events     = []

        for hl_time in hl_indices:
            i_hl = pos[hl_time]

            # at least 1 HH before this HL = uptrend context
            prev_hhs = [t for t in hh_indices if t < hl_time]
            if len(prev_hhs) < 1:
                continue

            hl_price = float(low.loc[hl_time])

            # first close below HL
            choch_time = None
            for j in range(i_hl + 1, min(i_hl + 1 + max_bars_after_hl, len(df))):
                if close.iloc[j] < hl_price:
                    choch_time = idx[j]
                    break
            if choch_time is None:
                continue

            choch_close = float(close.loc[choch_time])

            # all HLs broken by this close
            broken_hls = [
                t for t in hl_indices
                if t < choch_time and float(low.loc[t]) > choch_close
            ]
            latest_hl_broken = broken_hls[-1] if broken_hls else hl_time

            events.append({
                "type"            : "Bearish_CHoCH",
                "choch_time"      : choch_time,
                "hl_time"         : hl_time,
                "latest_hl_broken": latest_hl_broken,
                "hls_broken"      : len(broken_hls),
            })

        return pd.DataFrame(events) if events else pd.DataFrame()



    def _deduplicate_choch(self, bull: pd.DataFrame, bear: pd.DataFrame):
        import pandas as pd

        rows = []

        if not bull.empty:
            for _, r in bull.iterrows():
                rows.append({
                    "direction"  : "Bull",
                    "choch_time" : r["choch_time"],
                    "ref_time"   : r["latest_lh_broken"],
                    "breaks"     : r["lhs_broken"],
                })

        if not bear.empty:
            for _, r in bear.iterrows():
                rows.append({
                    "direction"  : "Bear",
                    "choch_time" : r["choch_time"],
                    "ref_time"   : r["latest_hl_broken"],
                    "breaks"     : r["hls_broken"],
                })

        if not rows:
            return []

        combined = (
            pd.DataFrame(rows)
            .sort_values(["choch_time", "breaks"], ascending=[True, False])
            .reset_index(drop=True)
        )

        # per choch_time + direction keep row with most breaks
        combined = (
            combined
            .groupby(["choch_time", "direction"], sort=False)
            .first()
            .reset_index()
            .sort_values("choch_time")
            .reset_index(drop=True)
        )

        # enforce alternation; on repeat direction keep stronger signal
        deduped  = []
        last_dir = None

        for _, row in combined.iterrows():
            d = row["direction"]
            if d == last_dir:
                if row["breaks"] >= deduped[-1]["breaks"]:
                    deduped[-1] = row
            else:
                deduped.append(row)
                last_dir = d

        return deduped


    def summarize_choch(self):
        bull    = self.detect_bullish_choch()
        bear    = self.detect_bearish_choch()
        deduped = self._deduplicate_choch(bull, bear)

        if not deduped:
            return "No CHoCH detected."

        lines = []
        for row in deduped:
            choch_ts = row["choch_time"].strftime("%d %b %Y %H:%M")
            ref_ts   = row["ref_time"].strftime("%d %b %Y %H:%M")
            breaks   = int(row["breaks"])

            if row["direction"] == "Bull":
                break_str = f"breaking {breaks} LHs" if breaks > 1 else "breaking LH"
                lines.append(
                    f"Bullish CHoCH on {choch_ts} {break_str} "
                    f"(most recent LH: {ref_ts})."
                )
            else:
                break_str = f"breaking {breaks} HLs" if breaks > 1 else "breaking HL"
                lines.append(
                    f"Bearish CHoCH on {choch_ts} {break_str} "
                    f"(most recent HL: {ref_ts})."
                )

        return " ".join(lines)

    def detect_choch(self):
        """
        CHoCH Up   : price was in downtrend (making LH/LL), then closes ABOVE last LH → bullish CHoCH
        CHoCH Down : price was in uptrend (making HH/HL), then closes BELOW last HL → bearish CHoCH
        """
        # df = self.data.copy()
        self.data['CHoCH_Bull'] = 0
        self.data['CHoCH_Bear'] = 0

        last_lh_price = None  # last Lower High level
        last_hl_price = None  # last Higher Low level
        in_downtrend  = False
        in_uptrend    = False

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            # Track last Lower High (resistance in downtrend)
            if row['Lower_High'] == 1:
                last_lh_price = row['High']
                in_downtrend  = True

            # Track last Higher Low (support in uptrend)
            if row['Higher_Low'] == 1:
                last_hl_price = row['Low']
                in_uptrend    = True

            # Bullish CHoCH: close breaks above last Lower High
            if in_downtrend and last_lh_price is not None:
                if row['Close'] > last_lh_price:
                    self.data.iloc[i, self.data.columns.get_loc('CHoCH_Bull')] = 1
                    last_lh_price = None  # reset — CHoCH fired
                    in_downtrend  = False

            # Bearish CHoCH: close breaks below last Higher Low
            if in_uptrend and last_hl_price is not None:
                if row['Close'] < last_hl_price:
                    self.data.iloc[i, self.data.columns.get_loc('CHoCH_Bear')] = 1
                    last_hl_price = None  # reset
                    in_uptrend    = False

    def _detect_patterns(self):
        self.data['Golden_Cross'] = ((self.data['SMA_20'] > self.data['SMA_50']) & (self.data['SMA_20'].shift() <= self.data['SMA_50'].shift())).astype(int)
        self.data['Death_Cross'] = ((self.data['SMA_20'] < self.data['SMA_50']) & (self.data['SMA_20'].shift() >= self.data['SMA_50'].shift())).astype(int) * -1

        # Add Double Bottom
        hl_indices = self.data[self.data['Higher_Low'] == 1].index
        double_bottom = [0] * len(self.data)
        if len(hl_indices) >= 2:
            for k in range(1, len(hl_indices)):
                low1 = self.data.at[hl_indices[k-1], 'Low']
                low2 = self.data.at[hl_indices[k], 'Low']
                if abs(low2 - low1) / low1 < 0.005:  # Within 0.5% tolerance
                    double_bottom[self.data.index.get_loc(hl_indices[k])] = 1.7  # Reliability ~70% from general studies
        self.data['Double_Bottom'] = double_bottom

    def detect_candlestick_patterns(self):
        o = self.data['Open']; h = self.data['High']; l = self.data['Low']; c = self.data['Close']
        range = h-l
        body = abs(c - o)
        upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
        lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
        avg_body = body.rolling(window=10).mean()
        self.data['Doji'] = np.where(body < (avg_body * 0.3), 0, 0)  # Neutral, ~50-60% reliability as reversal signal
        self.data['Hammer'] = np.where(
            (lower_shadow >= body * 2) & (upper_shadow <= body * 0.1) & (c > o) & (body <= 0.3 * range), 1.7, 0)  # ~70% bullish reliability
        self.data['Pin_bar'] = np.where(
            (lower_shadow >= body * 2.5) & (upper_shadow <= body * 0.66) & (c > o) & (lower_shadow >= 0.66 * range), 1.7, 0)  # ~70% bullish reliability
        self.data['Shooting_Star'] = np.where(
            (upper_shadow > body * 2) & (lower_shadow < body * 0.5) & (c < o), -1.7, 0)  # ~72% bearish from Evening Star analog
        prev_open, prev_close = o.shift(1), c.shift(1)
        prev_body = abs(prev_close - prev_open)

        o1, c1, o2, c2, o3, c3 = o.shift(2), c.shift(2), o.shift(1), c.shift(1), o, c
        h1, h2, h3 = h.shift(2), h.shift(1), h
        l1, l2, l3 = l.shift(2), l.shift(1), l
        self.data['Bullish_Engulfing'] = np.where(
            (c > o) & (prev_close < prev_open) &
            (c > prev_open) & (o < prev_close) & (body > prev_body) & (l<=l2) & (h>=h2), 2, 0)  # 73% from studies

        # vol1, vol2, vol3 = volume.shift(2), volume.shift(1), volume

        # Calculate body sizes
        body1, body2, body3 = abs(c1 - o1), abs(c2 - o2), abs(c3 - o3)

        # Basic Morning Star geometric conditions
        basic_morning_star_conditions = (
            (c1 < o1) &                          # First candle is bearish
            (body2 < body1 * 0.25) &             # Middle candle is small (tightened from 0.3 to 0.25)
            (body2 < avg_body * 0.5) &           # Middle candle is small relative to overall market
            (c3 > o3) &                          # Third candle is bullish
            (c3 > (c1 + o1) / 2) &              # Third candle closes above midpoint of first candle
            # (c2 < c1) &                          # Middle candle is lower than first
            (c3 >= o1)  &                          # Third candle closes above first candle's open
            (o3 <= c1)
        )

        # Volume conditions
        # avg_vol_first_two = (vol1 + vol2) / 2
        # strong_volume_spike = vol3 > 2.0 * avg_vol_first_two           # Strong volume confirmation
        # moderate_volume_spike = vol3 > 1.5 * avg_vol_first_two         # Moderate volume confirmation

        # Pattern strength conditions (body size requirements)
        strong_first_candle = body1 > 1.5 * avg_body                   # First candle is significantly large
        strong_third_candle = body3 > 1.5 * avg_body                   # Third candle is significantly large
        third_matches_first = body3 >= body1 * 0.8                     # Third candle matches first candle size
        basic_strong_third = body3 > 1.3 * avg_body                    # Your original strong confirmation

        # Tiered scoring system
        # =====================

        # Tier 4 (Score: 4.3) - Strongest signal
        # All conditions: strong bodies on candles 1 & 3, third matches first, strong volume spike
        tier_4_conditions = (
            basic_morning_star_conditions &
            strong_first_candle &
            strong_third_candle &
            third_matches_first
            # strong_volume_spike
        )

        # Tier 3 (Score: 3.3) - Strong signal
        # Strong third candle body + moderate volume spike
        tier_3_conditions = (
            basic_morning_star_conditions &
            basic_strong_third &
            # moderate_volume_spike &
            ~tier_4_conditions                   # Exclude tier 4 to avoid overlap
        )

        # Tier 2 (Score: 2.3) - Moderate signal
        # Basic pattern with moderate volume confirmation
        tier_2_conditions = (
            basic_morning_star_conditions &
            # moderate_volume_spike &
            ~tier_3_conditions &
            ~tier_4_conditions                   # Exclude higher tiers
        )

        # Tier 1 (Score: 1.3) - Weak signal (optional)
        # Basic pattern without volume confirmation
        tier_1_conditions = (
            basic_morning_star_conditions &
            ~tier_2_conditions &
            ~tier_3_conditions &
            ~tier_4_conditions
        )

        # Apply tiered scoring using np.where
        self.data['Morning_Star'] = np.where(
            tier_4_conditions, 4.3,
            np.where(
                tier_3_conditions, 3.3,
                np.where(
                    tier_2_conditions, 2.3,
                    np.where(
                        tier_1_conditions, 1.3,
                        0
                    )
                )
            )
        )


        m = self.data['MACD']; ms = self.data['MACD_signal']
        m1, ms1, m2, ms2 = m.shift(1), ms.shift(1), m, ms
        self.data['MACD_final_signal'] = np.where(
            (m2 > ms2) & (m1 < ms1) & (m2 > 0), 1.5, 0
        )

        # Add Evening Star (bearish, 3-candle)
        self.data['Evening_Star'] = np.where(
            (c1 > o1) & (body2 < body1 * 0.3) & (c3 < o3) & (c3 < (c1 + o1)/2) & (c2 > c1), -2.3, 0)  # ~75% reliability

    def generate_signals(self):
        signals = [0] * len(self.data)
        # print(self.data[self.data['Higher_Low'] == 1].index)
        hl_indices = self.data[self.data['Higher_Low'] == 1].index
        recent_break_idx = None
        scores = [0] * len(self.data)
        extend = [0] * len(self.data)
        for i in range(50, len(self.data)):
            confirmation_score = 0
            # if self.data['MACD'].iloc[i] > self.data['MACD_signal'].iloc[i]: score += 1
            # else: score -= 1
            # if self.data['RSI'].iloc[i] < 30: score += 1
            # elif self.data['RSI'].iloc[i] > 70: score -= 1
            if self.data['Break_Prev_Higher_Low'].iloc[i] == 1:
                recent_break_idx = i
            if recent_break_idx is not None:
                # Determine extended window
                extended_window = 5  # Default
                second_break_idx = None
                if len(hl_indices) >= 2:
                    prev_prev_hl_idx = hl_indices[-2] if len(hl_indices) > 1 else None
                    if prev_prev_hl_idx:
                        # Find if/where prev_prev HL is broken after recent_break_idx
                        for k in range(recent_break_idx + 1, len(self.data)):
                            if self.data.at[self.data.index[k], 'Close'] < self.data.at[prev_prev_hl_idx, 'Low']:
                                second_break_idx = k
                                extended_window = second_break_idx - recent_break_idx
                                break
                if second_break_idx is None:
                    # No second break: conservative hold
                    signals[i] = 0
                    continue

                for j in range(recent_break_idx, recent_break_idx + extended_window + 1):
                # if i - recent_break_idx <= extended_window:
                    # Pattern strengths...
                    # if self.data['Close'].iloc[j] > self.data['EMA_26'].iloc[j]: confirmation_score += 1
                    # else: confirmation_score += 0
                    # if self.data['Close'].iloc[j] < self.data['BB_lower'].iloc[j]: confirmation_score += 1
                    # elif self.data['Close'].iloc[j] > self.data['BB_upper'].iloc[j]: confirmation_score -= 1

                    hammer_strength = self.data['Hammer'].iloc[j] if self.data['Hammer'].iloc[j] > 0 else 0
                    engulf_strength = self.data['Bullish_Engulfing'].iloc[j] if self.data['Bullish_Engulfing'].iloc[j] > 0 else 0
                    # db_strength = self.data['Double_Bottom'].iloc[j] if self.data['Double_Bottom'].iloc[j] > 0 else 0
                    # doji_strength = self.data['Doji'].iloc[j] if self.data['Doji'].iloc[j] > 0 else 0
                    ms_strength = self.data['Morning_Star'].iloc[j] if self.data['Morning_Star'].iloc[j] > 0 else 0
                    es_strength = self.data['Evening_Star'].iloc[j]
                    # mac_strength = self.data['MACD_final_signal'].iloc[j] if self.data['MACD_final_signal'].iloc[j] > 0 else 0
                    confirmation_score += hammer_strength  # Higher weight for strong patterns
                    confirmation_score += engulf_strength
                    # confirmation_score += mac_strength
                    # confirmation_score += doji_strength  # Lower weight for neutral
                    confirmation_score += ms_strength  # Morning Star strong bullish
                    confirmation_score += es_strength  # Evening Star strong bearish (subtract for sell bias)
                    scores[j] = confirmation_score
                    if confirmation_score >= 1 :
                        signals[j] = 1
                        extend[i] = j - recent_break_idx
                        break

                    # if self.data['RSI'].iloc[i] < 30:
                    #     confirmation_score += 1  # Oversold ~70-80% reliability in reversals
                    # if self.data['MACD'].iloc[i] > self.data['MACD_signal'].iloc[i]:
                    #     confirmation_score += 1  # ~65% reliability
                    # if
                    # if extended_window > 4 :
                    #     multiplier = 2
                    # else : multiplier = 1
                # scores[i] += int(confirmation_score)  # Boost based on total weighted strength
            # signals[i] = 1 if scores[i] >= 4 else 0
        self.data['Scores'] = scores
        self.data['Signal'] = signals
        self.data['Extend_period'] = extend

    def score_break_confirmations(self, window=4):
        pattern_cols = ['Hammer', 'Bullish_Engulfing', 'Doji', 'Shooting_Star', 'Double_Bottom', 'Morning_Star', 'Evening_Star']
        break_scores = [0]*len(self.data)
        pattern_hits = ['']*len(self.data)
        leading_confirms = [None]*len(self.data)

        for i in range(len(self.data)):
            if self.data['Break_Prev_Higher_Low'].iloc[i] == 1:
                score, hits = 0, []
                lead_found = False
                for j in range(1, window+1):
                    f = i + j
                    if f >= len(self.data): break
                    row = self.data.iloc[f]
                    for col in pattern_cols:
                        if col in self.data.columns and row.get(col, 0) != 0:  # Handle negative for bearish
                            pat_strength = abs(row[col]) / 100.0
                            score += row[col] / 100.0  # Positive for bullish, negative for bearish
                            hits.append(f"{col}@{pat_strength:.2f} +{j}")
                            if not lead_found:
                                leading_confirms[i] = col
                                lead_found = True
                    if 'RSI' in self.data.columns and row['RSI'] < 30:
                        score += 0.75  # ~75% reliability
                        hits.append(f"RSI<30@+{j}")
                        if not lead_found:
                            leading_confirms[i] = 'RSI<30'
                            lead_found = True
                    if all(c in self.data.columns for c in ['MACD', 'MACD_signal']) and row['MACD'] > row['MACD_signal']:
                        score += 0.65  # ~65% reliability
                        hits.append(f"MACD>Signal@+{j}")
                        if not lead_found:
                            leading_confirms[i] = 'MACD>Signal'
                            lead_found = True
                break_scores[i] = score
                pattern_hits[i] = ', '.join(hits)
        self.data['Break_Buy_Score'] = break_scores
        self.data['Break_Buy_Confirmations'] = pattern_hits
        self.data['Break_Buy_Leading_Confirm'] = leading_confirms
        print("Confirmation scoring with leading pattern tracked complete.")

    def analyze_market(self):
        latest = self.data.iloc[-1]
        print(f"\nCurrent Price: {latest['Close']:.5f}")
        print(f"RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.6f}, ADX: {latest['ADX']:.2f}, ATR: {latest['ATR']:.6f}")
        if latest['Break_Prev_Higher_Low'] == 1:
            print("Previous Higher Low broken: BULLISH window triggered")
        else:
            print("Previous Higher Low NOT broken")

        print("\nMost recent break trigger and confirmation scores (if any):")
        breaks = self.data[self.data['Break_Prev_Higher_Low'] == 1].iloc[-5:]
        if not breaks.empty:
            for idx, row in breaks.iterrows():
                lead = row['Break_Buy_Leading_Confirm'] if pd.notna(row['Break_Buy_Leading_Confirm']) else "No leading confirm"
                print(f"Close: {row['Close']:.5f}, Score: {row['Break_Buy_Score']}, Leading Confirm: {lead}")
        else:
            print("No break events in last 5 periods.")

        print("\nLeading Patterns in Recent 20 Periods:")
        recent = self.data.tail(20)
        patterns = ['Hammer', 'Bullish_Engulfing', 'Doji', 'Double_Bottom', 'Golden_Cross', 'Morning_Star', 'Evening_Star']
        for pat in patterns:
            if pat in self.data.columns:
                pat_locations = recent[recent[pat] != 0].index
                if not pat_locations.empty:
                    print(f"{pat}: Formed at {', '.join([str(p) for p in pat_locations])}")

        print("\nRecent overall buy signals & pattern confirmations:")
        display_cols = ['Close','RSI','MACD','Signal','ADX','ATR','Prev_Higher_Low','Break_Prev_Higher_Low',
                        'Break_Buy_Score','Break_Buy_Confirmations','Break_Buy_Leading_Confirm']
        print(self.data[display_cols].tail(10))
        recent = self.data.tail(10)
        buy_prob = (recent['Signal'] == 1).sum() / len(recent) * 100
        print(f"\nBuy Signal Probability (Last 10 bars): {buy_prob:.1f}%")

        # Last higher low info
        hl_df = self.data[self.data['Higher_Low'] == 1]
        if not hl_df.empty:
            last_hl_idx = hl_df.index[-1]
            last_hl_value = self.data.at[last_hl_idx, 'Low']
            print(f"\nLast Higher Low occurred at {last_hl_idx} with value {last_hl_value:.5f}")
            if len(hl_df) > 1:
                prev_hl_idx = hl_df.index[-2]
                prev_hl_value = self.data.at[prev_hl_idx, 'Low']
                print(f"Previous Higher Low occurred at {prev_hl_idx} with value {prev_hl_value:.5f}")
        else:
            print("\nNo Higher Low detected in the data.")

        # Last higher high info
        hh_df = self.data[self.data['Higher_High'] == 1]
        if not hh_df.empty:
            last_hh_idx = hh_df.index[-1]
            last_hh_value = self.data.at[last_hh_idx, 'High']
            print(f"Last Higher High occurred at {last_hh_idx} with value {last_hh_value:.5f}")
            if len(hh_df) > 1:
                prev_hh_idx = hh_df.index[-2]
                prev_hh_value = self.data.at[prev_hh_idx, 'High']
                print(f"Previous Higher High occurred at {prev_hh_idx} with value {prev_hh_value:.5f}")
        else:
            print("No Higher High detected in the data.")

        # Recommended action based on latest signal
        latest_signal = latest['Signal']
        reasons = []
        if latest['MACD'] > latest['MACD_signal']:
            reasons.append("MACD above signal line (bullish momentum)")
        if latest['MACD'] < latest['MACD_signal']:
            reasons.append("MACD below signal line (bearish momentum)")
        if latest['RSI'] < 30:
            reasons.append("RSI oversold (<30)")
        if latest['RSI'] > 70:
            reasons.append("RSI overbought (>70)")
        if latest['Close'] > latest['SMA_20']:
            reasons.append("Price above SMA20 (uptrend)")
        if latest['Close'] < latest['SMA_20']:
            reasons.append("Price below SMA20 (downtrend)")
        if latest['Close'] < latest['BB_lower']:
            reasons.append("Price below BB lower (potential reversal up)")
        if latest['Close'] > latest['BB_upper']:
            reasons.append("Price above BB upper (potential reversal down)")
        if pd.notna(latest['Break_Prev_Higher_Low']) and latest['Break_Prev_Higher_Low'] == 1:
            reasons.append("Recent break of previous higher low (potential liquidity grab)")
        if latest_signal == 1:
            action = "Buy"
        elif latest_signal == -1:
            action = "Sell"
        else:
            action = "Hold"
        why = "Reasons: " + "; ".join(reasons) if reasons else "Neutral indicators."
        print(f"\nRecommended Action: {action} - {why}")

    # def plot_structure_and_patterns_plotly(self, symbol: str, periods: int) :
    #     """
    #     Build an interactive Plotly chart and return HTML (for Lovable/Render).
    #     """
    #     if self.data is None or self.data.empty:
    #         return None

    #     df = self.data.tail(periods).copy()

    #     price_range = df['High'].max() - df['Low'].min()
    #     offset = price_range * 0.01

    #     fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    #     # Candlesticks
    #     fig.add_trace(
    #         go.Candlestick(
    #             x=df.index,
    #             open=df['Open'],
    #             high=df['High'],
    #             low=df['Low'],
    #             close=df['Close'],
    #             name="Price"
    #         ),
    #         row=1, col=1
    #     )

    #     # Structure
    #     hh = df[df['Higher_High'] == 1]
    #     hl = df[df['Higher_Low'] == 1]
    #     ll = df[df['Lower_Low'] == 1]
    #     lh = df[df['Lower_High'] == 1]

    #     fig.add_trace(
    #         go.Scatter(
    #             x=hh.index,
    #             y=hh['High'] + 3*offset,
    #             mode='markers+text',
    #             text=['HH'] * len(hh),
    #             textposition='top center',
    #             marker=dict(color='blue', size=9, symbol='triangle-up'),
    #             name='Higher High'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=lh.index,
    #             y=lh['High'] + 1*offset,
    #             mode='markers+text',
    #             text=['LH'] * len(lh),
    #             textposition='top center',
    #             marker=dict(color='orange', size=9, symbol='triangle-down'),
    #             name='Lower High'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=hl.index,
    #             y=hl['Low'] - 1*offset,
    #             mode='markers+text',
    #             text=['HL'] * len(hl),
    #             textposition='bottom center',
    #             marker=dict(color='green', size=9, symbol='triangle-down'),
    #             name='Higher Low'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=ll.index,
    #             y=ll['Low'] - 3*offset,
    #             mode='markers+text',
    #             text=['LL'] * len(ll),
    #             textposition='bottom center',
    #             marker=dict(color='red', size=9, symbol='triangle-up'),
    #             name='Lower Low'
    #         ),
    #         row=1, col=1
    #     )

    #     # Patterns
    #     hammer = df[df['Hammer'] > 0]
    #     engulf = df[df['Bullish_Engulfing'] > 0]
    #     pinbar = df[df['Pin_bar'] > 0]
    #     mstar = df[df['Morning_Star'] > 0]
    #     estar = df[df['Evening_Star'] < 0]

    #     fig.add_trace(
    #         go.Scatter(
    #             x=hammer.index,
    #             y=hammer['Low'] - 5*offset,
    #             mode='markers',
    #             marker=dict(color='green', size=8, symbol='circle'),
    #             name='Hammer',
    #             hovertext=['Hammer'] * len(hammer),
    #             hoverinfo='text+x+y'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=engulf.index,
    #             y=engulf['Low'] - 7*offset,
    #             mode='markers',
    #             marker=dict(color='darkgreen', size=8, symbol='square'),
    #             name='Bull Engulfing',
    #             hovertext=['Bullish Engulfing'] * len(engulf),
    #             hoverinfo='text+x+y'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=pinbar.index,
    #             y=pinbar['Low'] - 9*offset,
    #             mode='markers',
    #             marker=dict(color='teal', size=8, symbol='diamond'),
    #             name='Pin Bar',
    #             hovertext=['Pin bar'] * len(pinbar),
    #             hoverinfo='text+x+y'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=mstar.index,
    #             y=mstar['Low'] - 11*offset,
    #             mode='markers',
    #             marker=dict(color='purple', size=10, symbol='star'),
    #             name='Morning Star',
    #             hovertext=['Morning Star'] * len(mstar),
    #             hoverinfo='text+x+y'
    #         ),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=estar.index,
    #             y=estar['High'] + 5*offset,
    #             mode='markers',
    #             marker=dict(color='red', size=10, symbol='x'),
    #             name='Evening Star',
    #             hovertext=['Evening Star'] * len(estar),
    #             hoverinfo='text+x+y'
    #         ),
    #         row=1, col=1
    #     )

    #     fig.update_layout(
    #         title=f"{symbol} – Structure & Patterns",
    #         xaxis_title="Time",
    #         yaxis_title="Price",
    #         xaxis_rangeslider_visible=False,
    #         hovermode='x unified'
    #     )

    #     # Return HTML snippet for embedding
    #     return fig.to_html(full_html=False)

    def plot_structure_and_patterns_plotly(self, symbol: str, periods: int):
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df = self.data.tail(periods).copy()

        price_range = df['High'].max() - df['Low'].min()
        base_offset = price_range * 0.015  # slightly larger base gap

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.85, 0.15],  # main chart + volume strip
            vertical_spacing=0.03
        )

        # ── Candlesticks ─────────────────────────────────────────────
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'],   close=df['Close'],
                name="Price",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                whiskerwidth=0.5,
            ),
            row=1, col=1
        )

        # ── Volume bars (row 2) ───────────────────────────────────────
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'],
                marker_color=colors, opacity=0.6,
                name='Volume', showlegend=False
            ),
            row=2, col=1
        )

        # ── Helper: stagger labels that land on the same x-index ─────
        def stagger_y(series_index, base_y_series, step, direction=1):
            """
            Returns a list of (x, y, text) with y nudged so nearby labels
            don't stack on top of each other.
            direction: +1 = push up, -1 = push down
            """
            positions = []
            last_x_positions = {}  # x_idx → list of y already used
            for xi, (idx, row_val) in enumerate(zip(series_index, base_y_series)):
                x_key = str(idx)[:10]  # group by date (YYYY-MM-DD)
                used = last_x_positions.get(x_key, [])
                # Start from base and keep nudging if too close
                y = row_val
                attempts = 0
                while any(abs(y - u) < step * 0.8 for u in used) and attempts < 6:
                    y += direction * step
                    attempts += 1
                used.append(y)
                last_x_positions[x_key] = used
                positions.append(y)
            return positions

        # ── Structure labels ──────────────────────────────────────────
        structure_cfg = [
            # (column, anchor, direction, offset_mult, color, symbol, label)
            ('Higher_High', 'High',  +1,  2.5, '#1565C0', 'triangle-up',   'HH'),
            ('Lower_High',  'High',  +1,  1.0, '#E65100', 'triangle-down', 'LH'),
            ('Higher_Low',  'Low',   -1,  1.0, '#2E7D32', 'triangle-up',   'HL'),
            ('Lower_Low',   'Low',   -1,  2.5, '#B71C1C', 'triangle-down', 'LL'),
        ]

        for col, anchor, direction, mult, color, sym, label in structure_cfg:
            sub = df[df[col] == 1]
            if sub.empty:
                continue
            base_y = sub[anchor] + direction * mult * base_offset
            y_vals = stagger_y(sub.index, base_y, base_offset * 1.2, direction)

            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=y_vals,
                    mode='markers+text',
                    text=[f'<b>{label}</b>'] * len(sub),
                    textposition='top center' if direction > 0 else 'bottom center',
                    textfont=dict(size=11, color=color),
                    marker=dict(color=color, size=10, symbol=sym,
                                line=dict(width=1, color='white')),
                    name=label,
                    legendgroup='structure',
                    legendgrouptitle_text='Structure' if label == 'HH' else None,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "Price: %{customdata:.2f}<extra></extra>"
                    ),
                    customdata=sub[anchor].values
                ),
                row=1, col=1
            )
            # ── Change of Character (CHoCH) ───────────────────────────────
        choch_bull = df[df['CHoCH_Bull'] == 1]
        choch_bear = df[df['CHoCH_Bear'] == 1]

        # Bullish CHoCH — green label above candle
        if not choch_bull.empty:
            fig.add_trace(go.Scatter(
                x=choch_bull.index,
                y=choch_bull['High'] + 4 * base_offset,
                mode='markers+text',
                text=['<b>CHoCH↑</b>'] * len(choch_bull),
                textposition='top center',
                textfont=dict(size=11, color='#00C853'),
                marker=dict(color='#00C853', size=12, symbol='star',
                            line=dict(width=1.5, color='white')),
                name='CHoCH Bullish',
                legendgroup='choch',
                legendgrouptitle_text='Change of Character',
                hovertemplate="<b>Bullish CHoCH</b><br>%{x|%Y-%m-%d}<br>Close: %{customdata:.2f}<extra></extra>",
                customdata=choch_bull['Close'].values
            ), row=1, col=1)

        # Bearish CHoCH — red label below candle
        if not choch_bear.empty:
            fig.add_trace(go.Scatter(
                x=choch_bear.index,
                y=choch_bear['Low'] - 4 * base_offset,
                mode='markers+text',
                text=['<b>CHoCH↓</b>'] * len(choch_bear),
                textposition='bottom center',
                textfont=dict(size=11, color='#D50000'),
                marker=dict(color='#D50000', size=12, symbol='star',
                            line=dict(width=1.5, color='white')),
                name='CHoCH Bearish',
                legendgroup='choch',
                hovertemplate="<b>Bearish CHoCH</b><br>%{x|%Y-%m-%d}<br>Close: %{customdata:.2f}<extra></extra>",
                customdata=choch_bear['Close'].values
            ), row=1, col=1)

        # Horizontal dashed line at the broken level for each CHoCH
        for idx, row_data in choch_bull.iterrows():
            fig.add_hline(
                y=row_data['Close'],
                line=dict(color='rgba(0,200,83,0.3)', width=1, dash='dash'),
                row=1, col=1
            )
        for idx, row_data in choch_bear.iterrows():
            fig.add_hline(
                y=row_data['Close'],
                line=dict(color='rgba(213,0,0,0.3)', width=1, dash='dash'),
                row=1, col=1
            )


        # ── Pattern markers below candles ────────────────────────────
        pattern_cfg = [
            # (column, condition, anchor, direction, offset_mult, color, symbol, label)
            ('Hammer',            '>0', 'Low',  -1,  4,  '#00897B', 'circle',   'Hammer'),
            ('Bullish_Engulfing', '>0', 'Low',  -1,  6,  '#1B5E20', 'square',   'Bull Engulf'),
            ('Pin_bar',           '>0', 'Low',  -1,  8,  '#006064', 'diamond',  'Pin Bar'),
            ('Morning_Star',      '>0', 'Low',  -1, 10,  '#6A1B9A', 'star',     'Morning ★'),
            ('Evening_Star',      '<0', 'High', +1,  4,  '#C62828', 'x',        'Evening ★'),
        ]

        for col, cond, anchor, direction, mult, color, sym, label in pattern_cfg:
            if cond == '>0':
                sub = df[df[col] > 0]
            else:
                sub = df[df[col] < 0]
            if sub.empty:
                continue

            y_vals = sub[anchor] + direction * mult * base_offset

            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=y_vals,
                    mode='markers+text',
                    text=[label[0]] * len(sub),   # single-char label next to marker
                    textposition='bottom center' if direction < 0 else 'top center',
                    textfont=dict(size=8, color=color),
                    marker=dict(color=color, size=10, symbol=sym,
                                line=dict(width=1, color='white')),
                    name=label,
                    legendgroup='patterns',
                    legendgrouptitle_text='Patterns' if label == 'Hammer' else None,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "Price: %{customdata:.2f}<extra></extra>"
                    ),
                    customdata=sub[anchor].values
                ),
                row=1, col=1
            )

        # ── Swing structure line (connect HH/HL pivots) ───────────────
    # ── Swing lines ───────────────────────────────────────────────

        # Uptrend swing: connects HH + HL pivots (blue dotted)
        up_pivots = pd.concat([
            df[df['Higher_High'] == 1][['High']].rename(columns={'High': 'price'}),
            df[df['Higher_Low']  == 1][['Low']].rename(columns={'Low':  'price'})
        ]).sort_index()

        if len(up_pivots) >= 2:
            fig.add_trace(go.Scatter(
                x=up_pivots.index, y=up_pivots['price'],
                mode='lines',
                line=dict(color='rgba(21, 101, 192, 0.35)', width=1.5, dash='dot'),
                name='Uptrend Swing', hoverinfo='skip'
            ), row=1, col=1)

        # Downtrend swing: connects LH + LL pivots (red dotted)
        down_pivots = pd.concat([
            df[df['Lower_High'] == 1][['High']].rename(columns={'High': 'price'}),
            df[df['Lower_Low']  == 1][['Low']].rename(columns={'Low':  'price'})
        ]).sort_index()

        if len(down_pivots) >= 2:
            fig.add_trace(go.Scatter(
                x=down_pivots.index, y=down_pivots['price'],
                mode='lines',
                line=dict(color='rgba(183, 28, 28, 0.35)', width=1.5, dash='dot'),
                name='Downtrend Swing', hoverinfo='skip'
            ), row=1, col=1)


        # ── Layout ────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{symbol}</b> — Market Structure & Candlestick Patterns"
                    "<br><span style='font-size:13px;font-weight:normal;color:gray'>"
                    "HH/HL = Uptrend | LH/LL = Downtrend</span>"
                ),
                font=dict(size=18)
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                groupclick='toggleitem',
                bgcolor='rgba(0,0,0,0.03)',
                bordercolor='rgba(0,0,0,0.15)',
                borderwidth=1,
                font=dict(size=11),
                tracegroupgap=8,
                x=1.01, y=1,
                xanchor='left',
                yanchor='top'
            ),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            margin=dict(l=60, r=160, t=80, b=40),
        )

        fig.update_yaxes(title_text="Price (₹)", row=1, col=1,
                        showgrid=True, gridcolor='rgba(0,0,0,0.06)')
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                        showgrid=False)
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(0,0,0,0.05)',
            tickformat='%b %d',
            row=1, col=1
        )
        fig.update_xaxes(row=2, col=1, title_text="Date")

        return fig.to_html(full_html=False)

    def plot_recent(self, symbol, periods=20):
        if self.data is None or self.data.empty:
            print("No data to plot.")
            return
        recent_data = self.data.tail(periods)
        fig, ax = plt.subplots(figsize=(12,6))
        # Manual candlestick plotting
        for idx, row in recent_data.iterrows():
            color = 'g' if row['Close'] >= row['Open'] else 'r'
            ax.vlines(idx, row['Low'], row['High'], color='black', linewidth=1)
            ax.vlines(idx, min(row['Open'], row['Close']), max(row['Open'], row['Close']), color=color, linewidth=3)
        ax.plot(recent_data.index, recent_data['SMA_20'], label='SMA20', alpha=0.7)
        # Annotate patterns and signals
        for i, row in recent_data.iterrows():
            if row['Hammer'] > 0:
                ax.annotate(f'Hammer ({row["Hammer"]})', (i, row['High']), xytext=(5,5), textcoords='offset points', color='g')
            if row['Bullish_Engulfing'] > 0:
                ax.annotate(f'Engulfing ({row["Bullish_Engulfing"]})', (i, row['High']), xytext=(5,5), textcoords='offset points', color='g')
            if row['Double_Bottom'] > 0:
                ax.annotate(f'Double Bottom ({row["Double_Bottom"]})', (i, row['High']), xytext=(5,5), textcoords='offset points', color='g')
            if row['Signal'] == 1:
                ax.scatter(i, row['Close'], marker='^', c='g')
            if row['Signal'] == -1:
                ax.scatter(i, row['Close'], marker='v', c='r')
        ax.set_title(f'{symbol} Recent {periods} Periods with Patterns')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def analyze_multiple_symbols(symbols, username=None, password=None, recent_signal_count=5):
    summary_data = []
    SYMBOL_EXCHANGE_MAP = {"MAXHEALTH": "NSE", "XAGUSD": "OANDA", "EURUSD": "FX", "GBPUSD": "FX", "USDJPY": "FX", "USDCHF": "FX"
    , "AUDUSD": "FX", "NZDUSD": "FX", "USDCAD": "FX", "EURJPY": "FX", "EURGBP": "FX", "EURCHF": "FX", "GBPJPY": "FX", "AUDJPY": "FX"
    , "AUDCAD": "FX", "AUDNZD": "FX", "EURAUD": "FX", "EURNZD": "FX", "GBPAUD": "FX", "GBPNZD": "FX", "CHFJPY": "FX", "CADJPY": "FX"
    , "NZDJPY": "FX", "EURUSD.OANDA": "OANDA", "GBPUSD.OANDA": "OANDA", "XAUUSD.OANDA": "OANDA", "XAGUSD.OANDA": "OANDA", "XAUUSD": "OANDA"
    , "XAGUSD": "OANDA", "GOLD": "COMEX", "SILVER": "COMEX", "CRUDEOIL": "NYMEX", "WTI": "NYMEX", "BRENT": "ICEEUR", "BTCUSD": "CRYPTO"
    , "ETHUSD": "CRYPTO", "BTCUSDT": "BINANCE", "ETHUSDT": "BINANCE", "SOLUSDT": "BINANCE", "XRPUSDT": "BINANCE", "SPX": "CBOE", "SPY": "AMEX"
    , "NDX": "NASDAQ", "QQQ": "NASDAQ", "DJI": "DJ", "NIFTY": "NSE", "BANKNIFTY": "NSE", "FINNIFTY": "NSE", "NIFTYIT": "NSE", "AAPL": "NASDAQ"
    , "MSFT": "NASDAQ", "GOOG": "NASDAQ", "GOOGL": "NASDAQ", "AMZN": "NASDAQ", "TSLA": "NASDAQ", "META": "NASDAQ", "NVDA": "NASDAQ"
    , "NFLX": "NASDAQ", "AMD": "NASDAQ", "INTC": "NASDAQ", "PYPL": "NASDAQ", "ADBE": "NASDAQ", "PEP": "NASDAQ", "COST": "NASDAQ", "BRK.B": "NYSE"
    , "JPM": "NYSE", "BAC": "NYSE", "XOM": "NYSE", "CVX": "NYSE", "KO": "NYSE", "PG": "NYSE", "DIS": "NYSE", "V": "NYSE", "MA": "NYSE"
    , "BP.": "LSE", "VOD": "LSE", "HSBA": "LSE", "RIO": "LSE", "BHP": "LSE", "SIE": "XETR", "BMW": "XETR", "ADS": "XETR", "BAS": "XETR"
    , "RELIANCE": "NSE", "TCS": "NSE", "INFY": "NSE", "HDFCBANK": "NSE", "ICICIBANK": "NSE", "KOTAKBANK": "NSE", "SBIN": "NSE"
    , "AXISBANK": "NSE", "BAJFINANCE": "NSE", "ITC": "NSE", "LT": "NSE", "ASIANPAINT": "NSE", "MARUTI": "NSE", "M&M": "NSE", "SUNPHARMA": "NSE"
    , "TITAN": "NSE", "ULTRACEMCO": "NSE", "GRSE": "NSE", "PAYTM": "NSE", "PFC": "NSE", "HIRECT": "BSE", "GODREJAGRO": "NSE"
    , "ETERNAL": "NSE", "OLAELEC": "NSE", "ASTERDM": "NSE", "RALLIS": "NSE", "PCBL": "NSE", "LINDEINDIA": "NSE"
    , "SUZLON": "NSE"
}

    patterns_list = [
        'Hammer', 'Bullish_Engulfing', 'Double_Bottom',
        'Doji', 'Morning_Star', 'Evening_Star'
    ]
    for symbol in symbols:
        print(f"\n\n===== Analyzing {symbol} =====")
        analyzer = ForexAnalyzer(username, password)
        exchange = SYMBOL_EXCHANGE_MAP.get(symbol.upper(), 'NASDAQ')
        analyzer.get_data(symbol, exchange=exchange)

        if analyzer.data is not None and not analyzer.data.empty and 'Close' in analyzer.data.columns:
            analyzer.calculate_indicators()
            analyzer.detect_gaps()
            analyzer.detect_explosive_gap_buys()
            analyzer.generate_signals()
            analyzer.score_break_confirmations(window=2)
            analyzer.analyze_market()
            # analyzer.plot(symbol)
            # analyzer.plot_recent(symbol)
            analyzer.data.to_csv(f'{symbol}_data.csv')
            print(f"Downloaded data for {symbol} to {symbol}_data.csv")
            latest = analyzer.data.iloc[-1]
            recent = analyzer.data.tail(5)
            signal = latest['Signal']
            action = "Buy" if signal == 1 else "Sell" if signal == -1 else "Hold"
            macd_range = f"{recent['MACD'].min():.2f}-{recent['MACD'].max():.2f}"
            rsi_range = f"{recent['RSI'].min():.2f}-{recent['RSI'].max():.2f}"
            ma_status = "uptrend" if latest['Close'] > latest['SMA_20'] else "downtrend"
            patterns_observed = []
            patterns_list = ['Hammer', 'Bullish_Engulfing', 'Doji', 'Double_Bottom', 'Golden_Cross', 'Morning_Star', 'Evening_Star']
            for pat in patterns_list:
                if pat in recent.columns and (recent[pat] != 0).any():
                    patterns_observed.append(pat)
            pattern_str = f" with {'/'.join(patterns_observed)} observed" if patterns_observed else ""
            reason = f"as last 5 periods show MACD {macd_range} {'bullish' if latest['MACD'] > latest['MACD_signal'] else 'bearish'}, RSI {rsi_range} {'momentum' if action == 'Buy' else 'neutral'}, and MAs indicating {ma_status}{pattern_str}."
            summary_data.append({
                'Symbol': symbol,
                'Action': action,
                'Reason': reason
            })


            recent_signals = analyzer.data[analyzer.data['Signal'] != 0].tail(recent_signal_count)
            print("Most Recent Buy Signals - ")
            print(recent_signals.iloc[:, 0])
            signal_summary = []
            for idx, row in recent_signals.iterrows():
                patterns_hit = [pat for pat in patterns_list if pat in row and row[pat] != 0]
                # score = row['Scores']
                sig_type = "Buy" if row['Signal'] > 0 else "Sell" if row['Signal'] < 0 else "Hold"
                signal_summary.append(f"Signal: {sig_type}, Patterns: {', '.join(patterns_hit) if patterns_hit else 'None'} at {idx}")

            last_signal_str = "\n".join(signal_summary) if signal_summary else "No signals found in last bars."

            # summary_data.append({
            #     'Symbol': symbol,
            #     'Action': action,
            #     'Reason': reason,
            #     'LastSignals': last_signal_str
            # })

        else:
            print(f"No usable data available for {symbol}.")
            summary_data.append({
                'Symbol': symbol,
                'Action': 'No Data',
                'Reason': 'No data fetched or missing Close column.'
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n\n===== Consolidated Summary for All Symbols =====\n")
        for _, row in summary_df.iterrows():
            print(f"**{row['Symbol']}**: {row['Action']} {row['Reason']}")


# if __name__ == "__main__":
#     symbols_list = ['XAUUSD', 'GBPUSD', 'AUDCAD']
#     analyze_multiple_symbols(symbols_list)

# ========== CLIENT INPUTS ==========
from typing import List, Optional, Any
import numpy as np
import pandas as pd

# TODO: import the real Interval and ForexAnalyzer from your own modules
# from your_module import Interval, ForexAnalyzer
API_BASE_URL = os.getenv("API_BASE_URL", "https://stock-analyzer-wguh.onrender.com")


TIMEFRAME_MAP = {
    "1m": Interval.in_1_minute,
    "3m": Interval.in_3_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "30m": Interval.in_30_minute,
    "45m": Interval.in_45_minute,
    "1h": Interval.in_1_hour,
    "2h": Interval.in_2_hour,
    "3h": Interval.in_3_hour,
    "4h": Interval.in_4_hour,
    "1D": Interval.in_daily,
    "1W": Interval.in_weekly,
    "1M": Interval.in_monthly,
}

exchange_map = {
    "EURUSD": "FX", "GBPUSD": "FX", "USDJPY": "FX", "USDCHF": "FX", "AUDUSD": "FX",
    "NZDUSD": "FX", "USDCAD": "FX", "EURJPY": "FX", "EURGBP": "FX", "EURCHF": "FX",
    "GBPJPY": "FX", "AUDJPY": "FX", "AUDCAD": "FX", "AUDNZD": "FX", "EURAUD": "FX",
    "EURNZD": "FX", "GBPAUD": "FX", "GBPNZD": "FX", "CHFJPY": "FX", "CADJPY": "FX",
    "NZDJPY": "FX", "EURUSD.OANDA": "OANDA", "GBPUSD.OANDA": "OANDA",
    "XAUUSD.OANDA": "OANDA", "XAGUSD.OANDA": "OANDA", "XAUUSD": "OANDA",
    "XAGUSD": "OANDA", "GOLD": "COMEX", "SILVER": "COMEX", "CRUDEOIL": "NYMEX",
    "WTI": "NYMEX", "BRENT": "ICEEUR", "BTCUSD": "CRYPTO", "ETHUSD": "CRYPTO",
    "BTCUSDT": "BINANCE", "ETHUSDT": "BINANCE", "SOLUSDT": "BINANCE",
    "XRPUSDT": "BINANCE", "SPX": "CBOE", "SPY": "AMEX", "NDX": "NASDAQ",
    "QQQ": "NASDAQ", "DJI": "DJ", "NIFTY": "NSE", "BANKNIFTY": "NSE",
    "FINNIFTY": "NSE", "NIFTYIT": "NSE", "AAPL": "NASDAQ", "MSFT": "NASDAQ",
    "GOOG": "NASDAQ", "GOOGL": "NASDAQ", "AMZN": "NASDAQ", "TSLA": "NASDAQ",
    "META": "NASDAQ", "NVDA": "NASDAQ", "NFLX": "NASDAQ", "AMD": "NASDAQ",
    "INTC": "NASDAQ", "PYPL": "NASDAQ", "ADBE": "NASDAQ", "PEP": "NASDAQ",
    "COST": "NASDAQ", "BRK.B": "NYSE", "JPM": "NYSE", "BAC": "NYSE", "XOM": "NYSE",
    "CVX": "NYSE", "KO": "NYSE", "PG": "NYSE", "DIS": "NYSE", "V": "NYSE",
    "MA": "NYSE", "BP.": "LSE", "VOD": "LSE", "HSBA": "LSE", "RIO": "LSE",
    "BHP": "LSE", "SIE": "XETR", "BMW": "XETR", "ADS": "XETR", "BAS": "XETR",
    "RELIANCE": "NSE", "TCS": "NSE", "INFY": "NSE", "HDFCBANK": "NSE",
    "ICICIBANK": "NSE", "KOTAKBANK": "NSE", "SBIN": "NSE", "AXISBANK": "NSE",
    "BAJFINANCE": "NSE", "ITC": "NSE", "LT": "NSE", "ASIANPAINT": "NSE",
    "MARUTI": "NSE", "M&M": "NSE", "SUNPHARMA": "NSE", "TITAN": "NSE",
    "ULTRACEMCO": "NSE", "GRSE": "NSE", "PAYTM": "NSE", "PFC": "NSE",
    "HIRECT": "BSE", "GODREJAGRO": "NSE", "MAXHEALTH": "NSE", "ETERNAL": "NSE",
    "OLAELEC": "NSE", "ASTERDM": "NSE", "RALLIS": "NSE", "PCBL": "NSE",
    "LINDEINDIA": "NSE", "SUZLON": "NSE",
}

# ---------- API MODELS ----------

class AnalyzeRequest(BaseModel):
    symbols: str          # e.g. "SUZLON" or "SUZLON,INFY"
    timeframe: str        # one of TIMEFRAME_MAP keys
    bars_to_show: int     # 10–2000 etc.

class SymbolResult(BaseModel):
    symbol: str
    start_datetime: Optional[str]
    trend_summary: str
    breach_summary: str
    choch_summary: str
    last_buy_signal: Optional[str]
    last_higher_low: Optional[str]
    last_higher_low_break: Optional[str]
    last_multi_hl_break: Optional[str]
    last_multi_hl_break_value: Optional[int]
    last_gap_up: Optional[str]
    last_explosive_gap_up: Optional[str]
    last_morning_stars: Optional[List[str]]
    csv_filename: Optional[str]
    chart_html: Optional[str]
    csv_url: Optional[str]   

class AnalyzeResponse(BaseModel):
    results: List[SymbolResult]
    message: str

# ---------- CORE ANALYSIS LOGIC (your code refactored into a function) ----------

def analyze_symbols(symbols: str, timeframe: str, bars_to_show: int) -> AnalyzeResponse:
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    interval_enum = TIMEFRAME_MAP.get(timeframe)

    if interval_enum is None:
        return AnalyzeResponse(results=[], message=f"Invalid timeframe: {timeframe}")

    results: List[SymbolResult] = []

    for symbol in symbol_list:
        analyzer = ForexAnalyzer()
        exchange = exchange_map.get(symbol.upper(), "FX")

        # Fetch data
        analyzer.get_data(
            symbol,
            exchange=exchange,
            interval=interval_enum,
            max_bars=int(bars_to_show),
            fetch_bars=5000
        )

        if analyzer.data is None or analyzer.data.empty:
            results.append(SymbolResult(
                symbol=symbol,
                start_datetime=None,
                trend_summary="No data fetched",
                breach_summary="No data fetched",
                choch_summary="No data fetched",
                last_buy_signal=None,
                last_higher_low=None,
                last_higher_low_break=None,
                last_multi_hl_break=None,
                last_multi_hl_break_value=None,
                last_gap_up=None,
                last_explosive_gap_up=None,
                last_morning_stars=None,
                csv_filename=None,
            ))
            continue

        # ---- your indicator logic unchanged ----
        analyzer.calculate_indicators()
        analyzer.detect_gaps()
        analyzer.detect_explosive_gap_buys()
        analyzer.generate_signals()
        analyzer.score_break_confirmations()
        analyzer.compute_primary_trend()
        analyzer.detect_choch()
        primary_text = analyzer.summarize_primary_trend()
        breach_text  = analyzer.summarize_breaches_in_primary_legs()
        choch_text = analyzer.summarize_choch()

        # print(primary_text)
        # if breach_text:
        #     print(breach_text)


        # save csv (optional, still on server disk)
        csv_name = f"{symbol}_data.csv"
        analyzer.data.to_csv(csv_name)
        csv_url = f"{API_BASE_URL}/download_csv?symbol={symbol}"

        chart_html = analyzer.plot_structure_and_patterns_plotly(symbol, int(bars_to_show))


        df = analyzer.data

        recent_signals = df[df["Signal"] != 0].tail(1)
        recent_HL = df[df["Higher_Low"] != 0].tail(1)
        recent_HL_break = df[df["Break_Prev_Higher_Low"] == 1].tail(1)
        recent_morning_star = df[df["Morning_Star"] != 0].tail(3)
        recent_multi_HL_break = df[df["Count_Higher_Low_Breaches"] != 0].tail(1).reset_index()
        recent_gap_up = df[df["Is_Gap_Up"] != 0].tail(1)
        recent_explosive_gap = df[df["Explosive_Gap_Buy"] != 0].tail(1)

        last_signal_dt = recent_signals.index[-1] if not recent_signals.empty else None
        last_HL_dt = recent_HL.index[-1] if not recent_HL.empty else None
        last_HL_break_dt = recent_HL_break.index[-1] if not recent_HL_break.empty else None
        recent_morning_star_dt = list(recent_morning_star.index) if not recent_morning_star.empty else None
        recent_gap_up_dt = recent_gap_up.index[-1] if not recent_gap_up.empty else None

        if recent_explosive_gap.empty:
            recent_explosive_gap_dt = None
        else:
            recent_explosive_gap_dt = recent_explosive_gap.index[-1]
            if recent_gap_up_dt is not None and recent_explosive_gap_dt > recent_gap_up_dt:
                recent_explosive_gap_dt = None

        if recent_multi_HL_break.empty:
            recent_multi_HL_break_dt = None
            recent_multi_HL_break_value = None
        else:
            recent_multi_HL_break_dt = recent_multi_HL_break["datetime"].iloc[0]
            recent_multi_HL_break_value = int(recent_multi_HL_break["Count_Higher_Low_Breaches"].iloc[0])

        start_datetime = analyzer.data.reset_index().head(1)["datetime"].iloc[0]
        start_datetime_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")

        res = SymbolResult(
            symbol=symbol,
            start_datetime=start_datetime_str,
            trend_summary=primary_text,
            breach_summary=str(breach_text) if breach_text is not None else None,
            choch_summary=str(choch_text) if choch_text is not None else None,
            last_buy_signal=str(last_signal_dt) if last_signal_dt is not None else None,
            last_higher_low=str(last_HL_dt) if last_HL_dt is not None else None,
            last_higher_low_break=str(last_HL_break_dt) if last_HL_break_dt is not None else None,
            last_multi_hl_break=str(recent_multi_HL_break_dt) if recent_multi_HL_break_dt is not None else None,
            last_multi_hl_break_value=recent_multi_HL_break_value,
            last_gap_up=str(recent_gap_up_dt) if recent_gap_up_dt is not None else None,
            last_explosive_gap_up=str(recent_explosive_gap_dt) if recent_explosive_gap_dt is not None else None,
            last_morning_stars=[str(x) for x in recent_morning_star_dt] if recent_morning_star_dt else None,
            csv_filename=csv_name,
            chart_html=chart_html,
            csv_url=csv_url,
        )

        results.append(res)

    return AnalyzeResponse(results=results, message="Analysis complete")

# ---------- API ENDPOINT ----------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest):
    return analyze_symbols(req.symbols, req.timeframe, req.bars_to_show)

@app.options("/analyze")
def options_analyze():
    return Response(status_code=200)

@app.get("/download_csv")
def download_csv(symbol: str):
    """
    Download the latest CSV generated for a symbol.
    Assumes analyze_symbols has already been called for that symbol.
    """
    csv_name = f"{symbol.upper()}_data.csv"
    if not os.path.exists(csv_name):
        return Response(status_code=404, content="CSV not found for this symbol")

    return FileResponse(
        path=csv_name,
        media_type="text/csv",
        filename=csv_name,
    )
