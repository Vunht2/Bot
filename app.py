import os
import logging
from datetime import datetime, timezone
from io import BytesIO
import base64

import requests
import pandas as pd
from flask import Flask, request, jsonify

# Import AIStrategy from strategy_ai.py
from strategy_ai import AIStrategy

# ========== Configuration ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(message)s")

# Telegram bot (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Bot parameters
BOT_CONFIG = {
    "interval": os.getenv("INTERVAL", "15m"),
    "size":     int(os.getenv("SIZE", "200")), # Kích thước dữ liệu nến để lấy
    "days":     float(os.getenv("DAYS", "1")), # Số ngày dữ liệu để lấy (nếu size không đủ)
}

# Bitget endpoints
BITGET_SPOT_URL = "https://api.bitget.com/api/v2/spot/public/market/candles"
BITGET_SWAP_URL = "https://api.bitget.com/api/swap/v3/market/history/kline"

# Initialize AI Strategy
ai_strategy = AIStrategy()
# Path to the trained AI model
AI_MODEL_PATH = 'ml_model.joblib'

# ========== Helpers ==========

def send_telegram_message(chat_id: str, text: str):
    """
    Gửi tin nhắn tới Telegram chat_id nếu đã cấu hình TELEGRAM_TOKEN.
    """
    if not TELEGRAM_TOKEN or not chat_id:
        logging.debug("Telegram not configured – skipping alert")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def fetch_bitget_klines(symbol: str,
                        period: str,
                        size: int = 100,
                        days: float = None,
                        chat_id: str = None) -> pd.DataFrame:
    """
    Lấy dữ liệu nến từ Bitget spot hoặc swap.
    Tự nhận biết spot vs swap qua hậu tố "_SPBL".
    """
    is_swap    = symbol.endswith("_SPBL")
    raw_symbol = symbol.replace("_SPBL", "") if is_swap else symbol
    url        = BITGET_SWAP_URL if is_swap else BITGET_SPOT_URL

    # Map period giữa Spot và Swap
    PERIOD_MAP = {
        "1m":  ("1min",  "1m"),
        "5m":  ("5min",  "5m"),
        "15m": ("15min", "15m"),
        "30m": ("30min", "30m"),
        "1h":  ("1hour", "1h"),
        "4h":  ("4hour", "4h"),
        "1d":  ("1day",  "1d"),
    }
    spot_p, swap_p = PERIOD_MAP.get(period, (period, period))
    real_period    = swap_p if is_swap else spot_p

    params = {"symbol": raw_symbol, "period": real_period, "size": size}
    if days:
        now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - int(days * 24 * 60 * 60 * 1000)
        params.update({"startTime": start_ms, "endTime": now_ms})

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"[HTTP ERROR] URL: {resp.url}, Status: {resp.status_code}, Body: {resp.text}")
        if chat_id:
            msg = (
                f"⚠️ *API Error* {resp.status_code}\n"
                f"URL: `{resp.url}`\n"
                f"Body: `{resp.text}`"
            )
            send_telegram_message(chat_id, msg)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"[REQUEST ERROR] Failed to fetch data: {e}")
        if chat_id:
            msg = f"⚠️ *Network Error*: `{e}`"
            send_telegram_message(chat_id, msg)
        return pd.DataFrame()

    payload = resp.json()
    rawdata = payload.get("data") or payload.get("datas") or []
    if not rawdata:
        logging.warning(f"No data returned for {symbol} ({real_period})")
        return pd.DataFrame()

    # Một số endpoint spot trả ít hơn 7 cột
    df = pd.DataFrame(rawdata)
    # Chuẩn hoá tên cột
    colmap = {
        0: "timestamp", 1: "open", 2: "high", 3: "low",
        4: "close", 5: "volume", 6: "quote_volume"
    }
    # Đảm bảo chỉ lấy các cột có trong colmap
    df = df.rename(columns=colmap).loc[:, [col for col in colmap.values() if col in df.columns]]

    # Chuyển đổi kiểu dữ liệu sang số và datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Chuyển đổi các cột giá và volume sang kiểu số, xử lý lỗi bằng cách chuyển thành NaN
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def create_chart_base64(df: pd.DataFrame, symbol: str) -> str:
    """
    Vẽ biểu đồ giá đơn giản, trả về string base64.
    """
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg") # Sử dụng backend không GUI
    fig, ax = plt.subplots(figsize=(10, 5)) # Tăng kích thước biểu đồ
    ax.plot(df["timestamp"], df["close"], color="blue", lw=1.5, label="Giá đóng cửa")
    ax.set_title(f"Biểu đồ giá {symbol}", fontsize=16)
    ax.set_xlabel("Thời gian", fontsize=12)
    ax.set_ylabel("Giá đóng cửa", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate() # Tự động định dạng ngày trên trục x
    plt.tight_layout() # Điều chỉnh layout cho phù hợp

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight") # Tăng DPI để ảnh nét hơn
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64

# ========== Flask App ==========

app = Flask(__name__)

# Load AI model on app startup
with app.app_context():
    if not ai_strategy.load_model(AI_MODEL_PATH):
        logging.warning("AI model not found or failed to load. AI predictions might not be available.")

@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Nhận vào JSON { "symbol": "BTCUSDT[_SPBL]" },
    trả về tín hiệu buy/sell/hold và dự đoán AI.
    """
    data = request.get_json(force=True)
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    df = fetch_bitget_klines(
        symbol,
        BOT_CONFIG["interval"],
        size=BOT_CONFIG["size"],
        days=BOT_CONFIG["days"],
        chat_id=CHAT_ID
    )
    if df.empty:
        return jsonify({"signal": None, "reason": "no data", "ai_prediction": None})

    # Simple price movement signal
    last_close = float(df["close"].iloc[-1])
    # Đảm bảo có ít nhất 2 cây nến để so sánh
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    signal = "buy" if last_close > prev_close else "sell" if last_close < prev_close else "hold"

    # AI Prediction
    ai_proba, ai_confidence = ai_strategy.predict_signal(df.copy()) # Truyền bản sao để tránh thay đổi df gốc
    ai_signal = "buy" if ai_proba >= 0.5 else "sell" # Ví dụ: nếu xác suất mua >= 0.5 thì là mua

    return jsonify({
        "symbol": symbol,
        "signal": signal, # Tín hiệu đơn giản dựa trên giá
        "price": last_close,
        "time": df["timestamp"].iloc[-1].isoformat(),
        "ai_prediction": {
            "probability_buy": round(ai_proba, 4),
            "confidence": ai_confidence,
            "ai_signal": ai_signal
        }
    })

@app.route("/chart", methods=["GET"])
def chart():
    """
    Trả về biểu đồ price chart dạng base64 PNG.
    Query string: ?symbol=BTCUSDT[_SPBL]
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol query param is required"}), 400

    df = fetch_bitget_klines(
        symbol,
        BOT_CONFIG["interval"],
        size=BOT_CONFIG["size"],
        days=BOT_CONFIG["days"]
    )
    if df.empty:
        return jsonify({"error": "no data"}), 404

    img_b64 = create_chart_base64(df, symbol)
    return jsonify({"symbol": symbol, "chart": img_b64})

@app.route("/train_ai", methods=["POST"])
def train_ai_model():
    """
    Endpoint để kích hoạt huấn luyện lại mô hình AI.
    Chỉ nên dùng cho mục đích phát triển/thử nghiệm.
    """
    try:
        # Lấy dữ liệu cho việc huấn luyện (ví dụ: BTCUSDT, ETHUSDT)
        # Bạn có thể điều chỉnh danh sách symbol và khoảng thời gian
        symbols_to_train = ["BTCUSDT", "ETHUSDT"]
        training_data_frames = []
        for symbol in symbols_to_train:
            df = fetch_bitget_klines(
                symbol,
                BOT_CONFIG["interval"],
                size=BOT_CONFIG["size"] * 5, # Lấy nhiều dữ liệu hơn để huấn luyện
                days=BOT_CONFIG["days"] * 5 # Lấy nhiều ngày hơn để huấn luyện
            )
            if not df.empty:
                training_data_frames.append(df)
            else:
                logging.warning(f"Could not fetch training data for {symbol}")

        if not training_data_frames:
            return jsonify({"status": "failed", "message": "No data available for training."}), 500

        ai_strategy.train_model(training_data_frames)
        return jsonify({"status": "success", "message": "AI model training initiated. Check logs for details."})
    except Exception as e:
        logging.error(f"Error during AI model training: {e}")
        return jsonify({"status": "failed", "message": f"Error during training: {e}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Trong môi trường phát triển cục bộ, bạn có thể chạy trực tiếp
    # Trong môi trường production, gunicorn sẽ quản lý việc chạy app
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True chỉ khi phát triển
