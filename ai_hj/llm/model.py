# models.py
import torch.nn as nn
from transformers import BertModel

class MultiTaskLegalBERT(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        hidden_size = self.bert.config.hidden_size
        
        # 수치 예측용 4가지 헤드
        self.win_rate_head = nn.Linear(hidden_size, 1)
        self.sentence_head = nn.Linear(hidden_size, 1)
        self.fine_head = nn.Linear(hidden_size, 1)
        self.risk_head = nn.Linear(hidden_size, 1)
        
        # 소송 분류
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask, labels=None, 
                win_rate=None, sentence=None, fine=None, risk=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # 각 수치 예측
        pred_win = self.win_rate_head(pooled_output).squeeze(-1)
        pred_sent = self.sentence_head(pooled_output).squeeze(-1)
        pred_fine = self.fine_head(pooled_output).squeeze(-1)
        pred_risk = self.risk_head(pooled_output).squeeze(-1)
        logits = self.classifier(pooled_output)
        
        loss = None
        if win_rate is not None:
            mse_fct = nn.MSELoss()
            loss = mse_fct(pred_win, win_rate) + \
                   mse_fct(pred_sent, sentence) + \
                   mse_fct(pred_fine, fine) + \
                   mse_fct(pred_risk, risk)
            
            loss_fct = nn.CrossEntropyLoss()
            loss += 0.1 * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {
            "loss": loss, 
            "win_rate": pred_win, 
            "sentence": pred_sent, 
            "fine": pred_fine, 
            "risk": pred_risk, 
            "logits": logits
        }
    
    def save_pretrained(self, save_path):
        """모델 저장 (Hugging Face 스타일)"""
        import os
        import torch
        os.makedirs(save_path, exist_ok=True)
        
        # 모델 가중치 저장
        torch.save(self.state_dict(), f"{save_path}/pytorch_model.bin")
        
        # BERT 설정 저장
        self.bert.config.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, model_path, num_labels=3):
        """저장된 모델 불러오기"""
        import torch
        
        # 모델 초기화
        model = cls("klue/bert-base", num_labels=num_labels)
        
        # 가중치 로드
        # state_dict = torch.load(f"{model_path}/pytorch_model.bin", 
        #                         map_location=torch.device('cpu'))
        
        # weights_only=False를 추가하여 모델을 정상적으로 로드합니다.
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", 
                                map_location=torch.device('cpu'),
                                weights_only=False)
        
        model.load_state_dict(state_dict)
        
        return model