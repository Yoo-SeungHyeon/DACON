import gc
import time
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline  # ✅ 파이프라인을 통해 메모리 효율적 처리

# 1. 데이터 불러오기
train_df = pd.read_csv("../data/important_train.csv")

# 2. Feature / Label 분리
y = train_df["Segment"]
X = train_df.drop(columns=["Segment"])

# 3. Label 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 인코더 및 스케일러 저장
joblib.dump(label_encoder, "label_encoder2.joblib")
joblib.dump(scaler, "scaler2.joblib")
print("✅ label_encoder 및 scaler 저장 완료")

# 6. Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=8432
)

# ✅ 7. 오버샘플링 + 언더샘플링 파이프라인 구성
# print("🔁 SMOTE + RandomUnderSampler 적용 중...")

# resampling_pipeline = Pipeline([
#     ('smote', SMOTE(random_state=42)),
#     ('under', RandomUnderSampler(random_state=42))
# ])

# X_train_res, y_train_res = resampling_pipeline.fit_resample(X_train, y_train)
# print(f"✅ 샘플링 완료. 학습 데이터 shape: {X_train_res.shape}, 클래스 분포: {pd.Series(y_train_res).value_counts().to_dict()}")

# 8. XGBoost 학습
depth_values = [6, 7, 8, 9, 10, 11, 12, 13]

for depth in depth_values:
    print(f"\n🌲 XGBoost max_depth={depth}") 
    model = XGBClassifier(max_depth=depth, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    start_time = time.time()
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"🔍 F1_macro score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"⏱ 학습 및 예측 시간: {elapsed_time:.2f}초")

    # 모델 저장
    model_filename = f"xgboost2_depth{depth}_f1{f1:.4f}.joblib"
    joblib.dump(model, model_filename)
    print(f"💾 모델 저장 완료: {model_filename}")

    # 메모리 해제
    del model
    gc.collect()
