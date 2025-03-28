import gc
import time
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# # 1. 중요 피처 리스트 불러오기
# important_features = joblib.load("important_features.pkl")
# print(f"✅ 중요 피처 {len(important_features)}개 로드 완료")

# 2. 새로운 학습 데이터셋 불러오기
train_df = pd.read_csv("../data/important_train.csv")

# Feature / Label 분리
y = train_df["Segment"]
X = train_df.drop(columns=["Segment"])

# Label 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 👉 인코더와 스케일러 저장
joblib.dump(label_encoder, "label_encoder2.joblib")
joblib.dump(scaler, "scaler2.joblib")
print("✅ label_encoder 및 scaler 저장 완료")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=8432)

depth_values = [6, 7, 8, 9, 10, 11, 12, 13]

for depth in depth_values:
    print(f"\n🌲 XGBoost max_depth={depth}") 
    model = XGBClassifier(max_depth=depth, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"🔍 F1_macro score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"⏱ 학습 및 예측 시간: {elapsed_time:.2f}초")

    # 모델 저장 (파일명에 max_depth와 f1 score 포함)
    model_filename = f"xgboost2_depth{depth}_f1{f1:.4f}.joblib"
    joblib.dump(model, model_filename)
    print(f"💾 모델 저장 완료: {model_filename}")

    # 메모리 해제를 위해 모델 변수 삭제 및 가비지 컬렉션 수행
    del model
    gc.collect()
