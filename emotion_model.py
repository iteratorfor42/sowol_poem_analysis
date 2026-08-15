"""
emotion_model.py
-----------------
KPoEM (Korean Poetry Emotion Mapping) 모델 래퍼.

한국학중앙연구원 디지털인문학연구소(AKS-DHLAB)가 공개한 감정 분류 모델을 사용.
KcELECTRA-base를 KOTE 데이터셋으로 1차 파인튜닝한 뒤, 근현대시 감정
라벨링 데이터셋인 KPoEM으로 2차(도메인 특화) 파인튜닝한 모델이다.

모델: https://huggingface.co/AKS-DHLAB/KPoEM (MIT License)
논문: LIM, I., Ji, H., & Kim, B. (2026). KPoEM: A Human-Annotated Dataset
      for Emotion Classification and RAG-Based Poetry Generation in
      Korean Modern Poetry. The Review of Korean Studies, 29(1), 161-206.
      https://doi.org/10.25024/review.2026.29.1.006

아래 클래스 구조는 모델 카드에 안내된 공식 사용 예시 코드를 그대로 따른다
(가중치 로드 방식, 43개 감정 라벨 순서를 임의로 바꾸면 분류 결과가 어긋남).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

REPO_ID = "AKS-DHLAB/KPoEM"


class KPoEMClassifier(nn.Module):
    """KPoEM 다중 라벨(multi-label) 감정 분류기."""

    # 모델 카드에 명시된 43개 감정 라벨. 순서가 학습 시 라벨 인덱스와 일치해야 함.
    LABELS = [
        "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔", "화남/분노", "존경",
        "기대감", "우쭐댐/무시함", "안타까움/실망", "비장함", "의심/불신", "뿌듯함", "편안/쾌적",
        "신기함/관심", "아껴주는", "부끄러움", "공포/무서움", "절망", "한심함", "역겨움/징그러움",
        "짜증", "어이없음", "없음", "패배/자기혐오", "귀찮음", "힘듦/지침", "즐거움/신남", "깨달음",
        "죄책감", "증오/혐오", "흐뭇함(귀여움/예쁨)", "당황/난처", "경악", "부담/안_내킴", "서러움",
        "재미없음", "불쌍함/연민", "놀람", "행복", "불안/걱정", "기쁨", "안심/신뢰",
    ]

    def __init__(self, repo_id: str = REPO_ID, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.electra = AutoModel.from_pretrained(repo_id)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.electra.config.hidden_size, len(self.LABELS)),
        )

        weights_path = hf_hub_download(repo_id=repo_id, filename="classifier_state.bin")
        self.classifier.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.to(self.device)
        self.eval()

    def forward(self, text: str) -> torch.Tensor:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.electra(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
            )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

    def analyze(self, text: str, threshold: float = 0.0) -> list[tuple[str, float]]:
        """
        텍스트를 입력받아 (감정라벨, 확률) 쌍의 리스트를 확률 내림차순으로 반환.
        threshold 이상인 감정만 반환 (기본값 0.0 -> 전체 43개 반환).
        """
        logits = self.forward(text)
        probabilities = torch.sigmoid(logits.squeeze())
        predictions = (probabilities > threshold).int()

        detected = []
        for i, is_detected in enumerate(predictions):
            if is_detected == 1:
                detected.append((self.LABELS[i], probabilities[i].item()))

        detected.sort(key=lambda x: x[1], reverse=True)
        return detected


if __name__ == "__main__":
    # 모델 카드의 예시와 동일한 문장으로 동작 확인
    classifier = KPoEMClassifier()
    example = "나의 생은 미친듯이 사랑을 찾아 헤매었으나"
    result = classifier.analyze(example, threshold=0.3)
    for label, prob in result:
        print(f"{label}: {prob:.4f}")
