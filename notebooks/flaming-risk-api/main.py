import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from deep_translator import GoogleTranslator
from contextlib import asynccontextmanager
import random

# --- グローバル変数の定義 ---
tokenizer = None
model = None
device = None
safe_v_all_norm = None
out_v_all_norm = None
safe_texts_sampled = []
out_texts_sampled = []
SAFE_P = None
OUT_P = None

# 各有害属性のサンプル後データを格納するリスト
out_severe_sampled = []
out_obscene_sampled = []
out_threat_sampled = []
out_insult_sampled = []
out_identity_sampled = []

# --- 1. アプリ起動時の準備（事前ロード） ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device, safe_v_all_norm, out_v_all_norm
    global safe_texts_sampled, out_texts_sampled, SAFE_P, OUT_P
    global out_severe_sampled, out_obscene_sampled, out_threat_sampled, out_insult_sampled, out_identity_sampled
    
    print("Initializing Model and Loading Cache...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=512)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    # 1. ベクトルデータのロード
    with open("data/bert_vectors_data.pkl", 'rb') as f:
        vector_data = pickle.load(f)
    safe_vec_all = vector_data['safe_vec_all']
    out_vec_all = vector_data['out_vec_all']
    
    # 2. テキスト・属性・パラメータデータのロード（★教えていただいたキー名に完全同期）
    with open("data/texts_label_data.pkl", 'rb') as f:
        cache_data = pickle.load(f)
        
    safe_txt_all = cache_data['safe_txt_all']
    out_txt_all = cache_data['out_txt_all']
    out_severe_all = cache_data['out_severe_all']
    out_obscene_all = cache_data['out_obscene_all']
    out_threat_all = cache_data['out_threat_all']
    out_insult_all = cache_data['out_insult_all']
    out_identity_all = cache_data['out_identity_all']
    
    # 信頼度計算用パラメータの取得
    SAFE_P = cache_data.get('safe_params', {'intercept': 0, 'coef_dif': 0, 'coef_most': 0})
    OUT_P = cache_data.get('out_params', {'intercept': 0, 'coef_dif': 0, 'coef_most': 0})

    # クラス不均衡の是正（アンダーサンプリング）
    min_size = min(len(safe_vec_all), len(out_vec_all))
    random.seed(1)
    safe_indices = random.sample(range(len(safe_vec_all)), min_size)
    out_indices = random.sample(range(len(out_vec_all)), min_size)
    
    # ベクトルのサンプリングとL2正規化
    safe_v_all = torch.tensor(np.array([safe_vec_all[i] for i in safe_indices])).to(device)
    out_v_all = torch.tensor(np.array([out_vec_all[i] for i in out_indices])).to(device)
    safe_v_all_norm = torch.nn.functional.normalize(safe_v_all, p=2, dim=1)
    out_v_all_norm = torch.nn.functional.normalize(out_v_all, p=2, dim=1)

    # テキストと各種属性のサンプリング
    safe_texts_sampled = [safe_txt_all[i] for i in safe_indices]
    out_texts_sampled = [out_txt_all[i] for i in out_indices]

    out_severe_sampled = [out_severe_all[i] for i in out_indices]
    out_obscene_sampled = [out_obscene_all[i] for i in out_indices]
    out_threat_sampled = [out_threat_all[i] for i in out_indices]
    out_insult_sampled = [out_insult_all[i] for i in out_indices]
    out_identity_sampled = [out_identity_all[i] for i in out_indices]
    
    print("Ready to serve!")
    yield

app = FastAPI(lifespan=lifespan, title="Flaming Risk Prediction API")

# --- 2. リクエストとレスポンスの型定義 ---
class RiskRequest(BaseModel):
    text: str

class RiskResponse(BaseModel):
    judge: str              # 判定結果 (SAFE / OUT)
    most_similar_text: str   # 最も類似した過去のテキスト
    severe: float           # 深刻な有害性
    obscene: float          # 猥褻性
    threat: float           # 脅迫性
    insult: float           # 侮辱性
    identity: float         # 差別性
    prob: float             # 信頼度（確率 %）

# --- 3. 推論エンドポイント ---
@app.post("/predict", response_model=RiskResponse)
def predict_risk(request: RiskRequest):
    try:
        # 日本語から英語への自動翻訳
        text_en = GoogleTranslator(source='ja', target='en').translate(request.text)
        
        with torch.no_grad():
            inputs = tokenizer(text_en, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            outputs = model(**inputs)
            
            # Mean Pooling & 正規化
            token_embeddings = outputs[0]
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_vector = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_vector_norm = torch.nn.functional.normalize(sentence_vector, p=2, dim=1)
            
            # コサイン類似度の計算
            safe_sims = torch.mm(sentence_vector_norm, safe_v_all_norm.T)
            out_sims = torch.mm(sentence_vector_norm, out_v_all_norm.T)
            
            ms_safe_val, idx_safe = torch.max(safe_sims, dim=1)
            ms_out_val, idx_out = torch.max(out_sims, dim=1)
            
            ms_safe = ms_safe_val.item()
            ms_out = ms_out_val.item()
            i_safe = idx_safe.item()
            i_out = idx_out.item()
            
            # 1. judge の決定
            judge = "SAFE" if ms_safe > ms_out else "OUT"
            
            # 2. most_similar_text および各種属性の抽出
            if judge == "SAFE":
                most_similar_text = safe_texts_sampled[i_safe]
                severe = 0.0
                obscene = 0.0
                threat = 0.0
                insult = 0.0
                identity = 0.0
                most_similarity = ms_safe
            else:
                most_similar_text = out_texts_sampled[i_out]
                severe = float(out_severe_sampled[i_out])
                obscene = float(out_obscene_sampled[i_out])
                threat = float(out_threat_sampled[i_out])
                insult = float(out_insult_sampled[i_out])
                identity = float(out_identity_sampled[i_out])
                most_similarity = ms_out
            
            # 3. 信頼度の算出
            dif_score = abs(ms_safe - ms_out)
            p = SAFE_P if judge == "SAFE" else OUT_P
            
            # ロジスティック回帰の計算
            logit = p['intercept'] + (p['coef_dif'] * dif_score) + (p['coef_most'] * most_similarity)
            prob_raw = 1 / (1 + np.exp(-float(logit)))
            prob = round(prob_raw * 100, 2)
            
            return RiskResponse(
                judge=judge,
                most_similar_text=most_similar_text,
                severe=severe,
                obscene=obscene,
                threat=threat,
                insult=insult,
                identity=identity,
                prob=prob
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
