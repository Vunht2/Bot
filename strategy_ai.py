import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib # Để lưu/tải mô hình
import numpy as np # Đảm bảo numpy được import

class AIStrategy:
    def __init__(self):
        self.model = None
        self.scaler = None # Có thể cần nếu bạn sử dụng StandardScaler (hiện tại không dùng)
        # Các đặc trưng mà mô hình sẽ học
        self.features = ['rsi', 'ma_cross_signal', 'divergence_signal', 'volume_change']

    def _calculate_indicators(self, df):
        """
        Tính toán các chỉ báo kỹ thuật cần thiết cho mô hình.
        """
        # Đảm bảo cột 'close' là numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Tính toán RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # Tính toán MA ngắn hạn và dài hạn
        df['ma_short'] = ta.trend.sma_indicator(close=df['close'], window=20)
        df['ma_long'] = ta.trend.sma_indicator(close=df['close'], window=50)

        # Tín hiệu giao cắt MA (1 nếu MA ngắn cắt lên MA dài, -1 nếu cắt xuống, 0 nếu không)
        df['ma_cross_signal'] = 0
        # Đảm bảo không có NaN trước khi so sánh
        # Tạo một bản sao để tránh SettingWithCopyWarning
        df_temp = df.dropna(subset=['ma_short', 'ma_long']).copy()
        df.loc[df_temp.index[df_temp['ma_short'] > df_temp['ma_long']], 'ma_cross_signal'] = 1
        df.loc[df_temp.index[df_temp['ma_short'] < df_temp['ma_long']], 'ma_cross_signal'] = -1


        # Thay đổi khối lượng
        df['volume_change'] = df['volume'].pct_change() * 100

        # Phát hiện phân kỳ RSI và gán tín hiệu
        # Hàm detect_rsi_divergence_v2 trả về list các phân kỳ
        divergences = self.detect_rsi_divergence_v2(df.copy()) # Truyền bản sao để tránh thay đổi df gốc
        df['divergence_signal'] = 0 # Mặc định không có phân kỳ

        if divergences:
            # Lấy phân kỳ gần nhất
            latest_divergence = divergences[0]
            # Gán tín hiệu dựa trên loại phân kỳ
            if latest_divergence['type'] == 'Bullish Divergence':
                # Gán tín hiệu phân kỳ tại cây nến cuối cùng hoặc gần cuối cùng
                # Kiểm tra nếu index của cây nến cuối cùng nằm trong khoảng của phân kỳ
                if latest_divergence['price_point2_idx'] == len(df) -1:
                    df.loc[len(df) - 1, 'divergence_signal'] = 1 # Tín hiệu mua
            elif latest_divergence['type'] == 'Bearish Divergence':
                if latest_divergence['price_point2_idx'] == len(df) -1:
                    df.loc[len(df) - 1, 'divergence_signal'] = -1 # Tín hiệu bán

        # Loại bỏ các hàng có giá trị NaN do tính toán chỉ báo
        # Chỉ loại bỏ các hàng có NaN trong các cột features
        df.dropna(subset=self.features, inplace=True)
        return df

    def detect_rsi_divergence_v2(self, df):
        """
        Phát hiện phân kỳ RSI (cả bullish và bearish) dựa trên giá đóng cửa và RSI.
        Phiên bản này tập trung vào sự hình thành đáy/đỉnh mới.
        """
        divergences = []
        # Đảm bảo df đã có cột 'rsi' và 'close'
        if 'rsi' not in df.columns or 'close' not in df.columns:
            # Nếu chưa có, tính toán lại các chỉ báo
            df = self._calculate_indicators(df.copy())
            if 'rsi' not in df.columns or 'close' not in df.columns: # Kiểm tra lại sau khi tính
                print("Lỗi: Không thể tính toán RSI hoặc giá đóng cửa.")
                return []

        # Cần ít nhất một số lượng nến nhất định để có đủ dữ liệu cho phân tích đỉnh/đáy đáng tin cậy
        if len(df) < 50: # Giảm ngưỡng tối thiểu để linh hoạt hơn
            return []

        # Chuyển đổi Series thành list hoặc numpy array để dễ xử lý theo chỉ mục
        closes = df['close'].values
        rsis = df['rsi'].values
        dates = df.index.values # Sử dụng index của DataFrame làm ngày

        # Tìm các điểm swing (đỉnh/đáy cục bộ) cho giá và RSI
        # Phương pháp đơn giản: so sánh với 2 điểm lân cận
        # Để tránh lỗi out of bounds, sử dụng slice [1:-1] và cộng 1 vào chỉ mục kết quả
        price_peaks_idx = np.where((closes[1:-1] > closes[0:-2]) & (closes[1:-1] > closes[2:]))[0] + 1
        rsi_peaks_idx = np.where((rsis[1:-1] > rsis[0:-2]) & (rsis[1:-1] > rsis[2:]))[0] + 1

        price_troughs_idx = np.where((closes[1:-1] < closes[0:-2]) & (closes[1:-1] < closes[2:]))[0] + 1
        rsi_troughs_idx = np.where((rsis[1:-1] < rsis[0:-2]) & (rsis[1:-1] < rsis[2:]))[0] + 1


        # --- Bullish Divergence (Regular) ---
        # Giá tạo đáy thấp hơn, RSI tạo đáy cao hơn
        for i in range(len(price_troughs_idx) - 1):
            trough1_idx_abs = price_troughs_idx[i]
            trough2_idx_abs = price_troughs_idx[i+1]

            # Đảm bảo có đủ khoảng cách giữa các đáy (ví dụ: ít nhất 3 nến)
            if trough2_idx_abs - trough1_idx_abs < 3:
                continue

            # Kiểm tra đáy giá: Giá hiện tại thấp hơn giá trước
            if closes[trough2_idx_abs] < closes[trough1_idx_abs]:
                # Kiểm tra đáy RSI: RSI hiện tại cao hơn RSI trước (trong cùng khoảng thời gian)
                # Tìm đáy RSI gần nhất sau trough1_idx_abs và trước trough2_idx_abs
                rsi_troughs_in_range = [idx for idx in rsi_troughs_idx if trough1_idx_abs < idx < trough2_idx_abs]
                if not rsi_troughs_in_range:
                    continue

                # Lấy đáy RSI cuối cùng trong khoảng thời gian giữa hai đáy giá
                rsi_trough1_candidate_idx_abs = max(rsi_troughs_in_range)

                if rsis[trough2_idx_abs] > rsis[rsi_trough1_candidate_idx_abs]:
                    divergences.append({
                        'type': 'Bullish Divergence',
                        'price_point1_idx': trough1_idx_abs,
                        'price_point2_idx': trough2_idx_abs,
                        'rsi_point1_idx': rsi_trough1_candidate_idx_abs,
                        'rsi_point2_idx': trough2_idx_abs,
                        'date1': dates[trough1_idx_abs],
                        'date2': dates[trough2_idx_abs],
                        'price1': closes[trough1_idx_abs],
                        'price2': closes[trough2_idx_abs],
                        'rsi1': rsis[rsi_trough1_candidate_idx_abs],
                        'rsi2': rsis[trough2_idx_abs],
                        'last_candle_date': dates[-1] # Ngày của cây nến cuối cùng
                    })


        # --- Bearish Divergence (Regular) ---
        # Giá tạo đỉnh cao hơn, RSI tạo đỉnh thấp hơn
        for i in range(len(price_peaks_idx) - 1):
            peak1_idx_abs = price_peaks_idx[i]
            peak2_idx_abs = price_peaks_idx[i+1]

            # Đảm bảo có đủ khoảng cách giữa các đỉnh (ví dụ: ít nhất 3 nến)
            if peak2_idx_abs - peak1_idx_abs < 3:
                continue

            # Kiểm tra đỉnh giá: Giá hiện tại cao hơn giá trước
            if closes[peak2_idx_abs] > closes[peak1_idx_abs]:
                # Kiểm tra đỉnh RSI: RSI hiện tại thấp hơn RSI trước (trong cùng khoảng thời gian)
                rsi_peaks_in_range = [idx for idx in rsi_peaks_idx if peak1_idx_abs < idx < peak2_idx_abs]
                if not rsi_peaks_in_range:
                    continue

                rsi_peak1_candidate_idx_abs = max(rsi_peaks_in_range)

                if rsis[peak2_idx_abs] < rsis[rsi_peak1_candidate_idx_abs]:
                    divergences.append({
                        'type': 'Bearish Divergence',
                        'price_point1_idx': peak1_idx_abs,
                        'price_point2_idx': peak2_idx_abs,
                        'rsi_point1_idx': rsi_peak1_candidate_idx_abs,
                        'rsi_point2_idx': peak2_idx_abs,
                        'date1': dates[peak1_idx_abs],
                        'date2': dates[peak2_idx_abs],
                        'price1': closes[peak1_idx_abs],
                        'price2': closes[peak2_idx_abs],
                        'rsi1': rsis[rsi_peak1_candidate_idx_abs],
                        'rsi2': rsis[peak2_idx_abs],
                        'last_candle_date': dates[-1]
                    })

        # Chỉ lấy phân kỳ gần nhất (tại nến cuối cùng hoặc rất gần cuối cùng)
        # Lấy index của cây nến cuối cùng trong df
        last_candle_abs_idx = len(df) - 1

        # Lọc các phân kỳ mà điểm thứ 2 (price_point2_idx) nằm trong 5 nến cuối cùng
        recent_divergences = [
            d for d in divergences
            if d['price_point2_idx'] >= last_candle_abs_idx - 5
        ]

        # Nếu có nhiều phân kỳ gần nhất, ưu tiên phân kỳ có 'price_point2_idx' gần cuối cùng nhất
        if recent_divergences:
            # Sắp xếp để lấy phân kỳ mới nhất theo price_point2_idx
            recent_divergences.sort(key=lambda x: x['price_point2_idx'], reverse=True)
            return [recent_divergences[0]] # Trả về phân kỳ gần nhất
        return []


    def _generate_labels(self, df, future_candles=5, price_increase_threshold=0.01):
        """
        Tạo nhãn cho dữ liệu: 1 nếu giá tăng X% trong Y nến tiếp theo, 0 nếu không.
        """
        # Đảm bảo các cột là numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Shift future_max_close before dropping NaNs to align correctly
        # Tính toán giá cao nhất trong 'future_candles' nến tiếp theo
        df['future_max_close'] = df['close'].shift(-future_candles).rolling(window=future_candles).max()
        # Tạo nhãn: 1 nếu giá tăng vượt ngưỡng, 0 nếu không
        df['label'] = ((df['future_max_close'] - df['close']) / df['close'] > price_increase_threshold).astype(int)

        # Loại bỏ các hàng có giá trị NaN do shift/rolling
        df.dropna(subset=['label'], inplace=True)
        return df

    def train_model(self, data_frames, future_candles=5, price_increase_threshold=0.01):
        """
        Huấn luyện mô hình Machine Learning.
        data_frames là một list các DataFrame (từ nhiều cặp tiền tệ hoặc nhiều khoảng thời gian)
        """
        all_data = pd.DataFrame()
        for df in data_frames:
            # Đảm bảo df có đủ dữ liệu cho _calculate_indicators và _generate_labels
            # Cần ít nhất 50 nến để tính toán chỉ báo và 5 nến cho future_candles
            if len(df) < 55:
                print(f"Bỏ qua DataFrame với {len(df)} nến, không đủ để huấn luyện.")
                continue

            df_processed = self._calculate_indicators(df.copy())
            df_labeled = self._generate_labels(df_processed.copy(), future_candles, price_increase_threshold)
            all_data = pd.concat([all_data, df_labeled], ignore_index=True)

        # Đảm bảo không có NaN trong các cột đặc trưng và nhãn trước khi huấn luyện
        all_data.dropna(subset=self.features + ['label'], inplace=True)

        if all_data.empty:
            print("Không đủ dữ liệu để huấn luyện mô hình sau khi xử lý và gắn nhãn.")
            return

        X = all_data[self.features]
        y = all_data['label']

        # Xử lý trường hợp chỉ có một lớp trong y (ví dụ: tất cả là 0 hoặc tất cả là 1)
        if y.nunique() < 2:
            print(f"Chỉ có một lớp trong dữ liệu huấn luyện (tất cả là {y.iloc[0]}). Không thể huấn luyện mô hình.")
            return

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # RandomForestClassifier thường hoạt động tốt với dữ liệu tài chính
        # 'balanced' để xử lý mất cân bằng lớp (nếu số lượng mẫu mua và bán không đều)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)

        # Đánh giá mô hình
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, zero_division=0)) # zero_division=0 để tránh lỗi khi có lớp không có mẫu

        # Lưu mô hình đã huấn luyện
        joblib.dump(self.model, 'ml_model.joblib')
        print("Mô hình đã được huấn luyện và lưu vào ml_model.joblib")

    def load_model(self, model_path='ml_model.joblib'):
        """Tải mô hình đã huấn luyện."""
        try:
            self.model = joblib.load(model_path)
            print(f"Mô hình đã được tải từ {model_path}")
            return True
        except FileNotFoundError:
            print(f"Không tìm thấy mô hình tại {model_path}. Cần huấn luyện mô hình trước.")
            self.model = None
            return False
        except Exception as e:
            print(f"Lỗi khi tải mô hình từ {model_path}: {e}. Có thể file bị hỏng hoặc không tương thích.")
            self.model = None
            return False


    def predict_signal(self, df):
        """
        Dự đoán tín hiệu mua cho cây nến cuối cùng.
        Trả về xác suất của lớp 1 (mua) và độ tin cậy.
        """
        if self.model is None:
            return 0.0, "Mô hình chưa sẵn sàng"

        df_processed = self._calculate_indicators(df.copy())

        if df_processed.empty:
            return 0.0, "Không đủ dữ liệu để dự đoán sau khi tính chỉ báo."

        # Lấy đặc trưng của cây nến cuối cùng
        # Kiểm tra xem df_processed có đủ hàng không
        if len(df_processed) == 0:
            return 0.0, "Không đủ dữ liệu sau khi xử lý chỉ báo."

        last_candle_features = df_processed[self.features].iloc[-1].values.reshape(1, -1)

        # Kiểm tra NaN trong features_for_prediction
        if np.isnan(last_candle_features).any():
            print("Cảnh báo: Đặc trưng cho dự đoán chứa giá trị NaN. Không thể dự đoán.")
            return 0.0, "Dữ liệu không đầy đủ cho dự đoán"

        try:
            # Xác suất của lớp 1 (mua)
            prediction_proba = self.model.predict_proba(last_candle_features)[0][1]

            # Gán độ tin cậy dựa trên xác suất
            if prediction_proba >= 0.7:
                confidence = "strong"
            elif prediction_proba >= 0.55:
                confidence = "medium"
            else:
                confidence = "weak"

            return prediction_proba, confidence
        except Exception as e:
            print(f"Lỗi trong quá trình dự đoán ML: {e}")
            return 0.0, f"Lỗi dự đoán: {e}"
