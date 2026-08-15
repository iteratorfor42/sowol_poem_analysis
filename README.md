# 김소월 『진달래꽃』 감정 분석 (KPoEM)

김소월 시집 『진달래꽃』 중 "바다가 변하야 뽕나무밭 된다고" 섹션에 실린
9편을, 한국학중앙연구원 디지털인문학연구소(AKS-DHLAB)가 공개한 KPoEM
(Korean Poetry Emotion Mapping) 모델로 감정 분석하는 파이프라인입니다.

효경언해 OCR 프로젝트(`ocr_hyogyeong`)와 이어지는 작업으로, 
"비정형텍스트 → 구조화된 데이터 → 정량적 해석"이라는 같은 흐름에서
후반부(정량적 해석) 단계에 해당합니다.

## 0. 이 프로젝트가 효경언해 프로젝트와 다른 점: 저작권

김소월 시인(1902~1934)은 사망한 지 70년이 훨씬 지났습니다. 
한국 저작권법상 저작재산권 보호기간은 저작자 사후 70년까지이므로, 
『진달래꽃』 원문은 현재 **퍼블릭 도메인**입니다. 

그래서 효경언해 프로젝트 때와 달리:

- 시 원문을 `data/raw/jindallaekkot_9poems.json`에 그대로 저장해 두었고
- 이 저장소를 그대로 git에 올리고 공개해도 원문 자체는 저작권 문제가 없습니다
(다만 KPoEM 모델·데이터셋 자체의 라이선스/인용 요구사항은 별개이니
아래 "출처 및 인용" 섹션을 참고하세요.)
- 시 원문은 `https://ko.wikisource.org/wiki/%EC%A7%84%EB%8B%AC%EB%9E%98%EA%BD%83_(%EC%8B%9C%EC%A7%91)`에서 가져왔습니다.
(`https://ko.wikisource.org/wiki/진달래꽃_(시집)`으로도 접속 가능합니다.)



## 1. 설치

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 환경 관련 참고

이 프로젝트는 `transformers`/`torch`가 **필수** 의존성입니다 
(효경언해 프로젝트에서는 선택 사항이었지만, 
KPoEM 모델 자체가 이 두 라이브러리 위에서 동작하기 때문입니다). 
Python 3.14처럼 아직 `torch`가 정식 지원하지 않는 버전을 쓰고 있다면 설치가 실패할 수 있습니다. 

이 경우:

- `python --version`으로 현재 버전 확인
- Python 3.11~3.12 등 `torch`가 지원하는 버전으로 별도 가상환경을 만들어
  그 안에서 `pip install -r requirements.txt` 재시도


## 2. 실행

```bash
python main.py \
    --json data/raw/jindallaekkot_9poems.json \
    --threshold 0.3 \
    --out-json data/output/emotion_analysis.json \
    --out-heatmap data/output/emotion_heatmap.png \
    --out-bar data/output/top_emotions_per_poem.png
```

첫 실행 시 KPoEM 모델 가중치를 Hugging Face에서 내려받기 때문에 
시간이 걸릴 수 있습니다 (이후에는 로컬 캐시 사용).

- `--threshold` : 감정을 "검출됨"으로 판단하는 확률 기준값 (기본 0.3)
- `--no-lines` : 행 단위 분석을 생략하고 시 전체 단위로만 분석 (속도 ↑, 세밀함 ↓)
- `--top-n-emotions` : 히트맵에 표시할 감정 개수 (기본 12개, 43개 전부 넣으면 과밀해짐)

실행 결과:
1. `data/output/emotion_analysis.json` — 시-행-감정이 연결된 구조화된 결과
2. `data/output/emotion_heatmap.png` — 9편 x 상위 감정 히트맵
3. `data/output/top_emotions_per_poem.png` — 시별 top5 감정 막대그래프
4. 콘솔에 시별 top3 감정 요약 출력

## 3. 파이프라인 구성

효경언해 프로젝트의 4단계 구조를 텍스트 분석에 맞게 재구성했습니다.

| 효경언해 (OCR) | 이 프로젝트 (감정 분석) |
|---|---|
| 전처리 (이진화/노이즈제거) | `preprocess.py` — 시 로드/텍스트 정제 |
| 레이아웃 분석 + 문자 인식 | `emotion_model.py` — KPoEM 감정 분류 |
| (구조화) | `analyze.py` — 시/행 단위 결과를 구조화된 레코드로 변환 |
| 후처리 | `visualize.py` — 히트맵/막대그래프로 시각화 |

## 4. 모델 설명: KPoEM

- **베이스 모델**: KcELECTRA-base
- **파인튜닝**: 범용 감정 데이터셋 KOTE로 1차 파인튜닝 → 근현대시 특화
  감정 라벨링 데이터셋 KPoEM으로 2차(도메인) 파인튜닝
- **출력**: 43개 감정 라벨에 대한 다중 라벨(multi-label) 확률
- **주의**: 이 모델은 한국 근현대시(김소월류 서정시 등)의 은유적·정서적 표현에 맞춰 파인튜닝된 모델입니다. 
  일반 문어체나 다른 장르(구술 문학,  현대 산문시 등)에도 그대로 잘 맞는다는 보장은 없습니다 
  — 결과를 그대로 "정답"으로 받아들이기보다, 
  정성적 해석과 교차 검증하는 자료로 쓰는 것을 권장합니다.

### 감정 분석 단위에 대한 메모

시 전체를 한 번에 넣으면 512 토큰 제한 안에서 여러 정서가 뭉뚱그려질 수 있어, 
이 파이프라인은 **시 전체 단위**와 **행 단위**를 모두 분석해 JSON에 같이 저장합니다. 
어느 단위가 "옳은" 분석 단위인지는 정해진 답이 없고,
연구 목적에 따라 달라질 수 있는 지점입니다 
(Session 03 세션 소개에서 언급된 "행 vs 연 vs 시 전체" 단위 문제와 같은 맥락).

## 5. 출처 및 인용

- 모델: [AKS-DHLAB/KPoEM](https://huggingface.co/AKS-DHLAB/KPoEM) (MIT License)
- 데이터셋: [AKS-DHLAB/KPoEM dataset](https://huggingface.co/datasets/AKS-DHLAB/KPoEM),
           [Zenodo](https://zenodo.org/records/15572285)
- 파인튜닝 소스코드: [GitHub - AKS-DHLAB/KPoEM](https://github.com/AKS-DHLAB/KPoEM)
- 논문:
  > LIM, I., Ji, H., & Kim, B. (2026). KPoEM: A Human-Annotated Dataset for
  > Emotion Classification and RAG-Based Poetry Generation in Korean Modern
  > Poetry. *The Review of Korean Studies, 29*(1), 161-206.
  > https://doi.org/10.25024/review.2026.29.1.006

이 저장소의 `emotion_model.py`는 위 모델 카드에 공식으로 안내된 사용예시 
코드(가중치 로드 방식, 43개 라벨 순서 포함)를 그대로 따릅니다 
— 라벨 순서를 임의로 바꾸면 분류 결과가 어긋나므로 수정하지 않는 것을 권장합니다.

## 6. 다음 단계

1. 형태소 분석(예: `kiwipiepy`)을 추가해, 
   감정 분석 결과와 어휘 패턴을 함께 보는 방향으로 확장
2. 김소월 다른 시집·다른 시인과의 비교로 확장해 "시대·시인별 문체 비교" 축으로 발전
3. 히트맵 대신 UMAP/t-SNE로 감정 벡터를 2차원에 투영해 
   시들 사이의 근접성을 다른 방식으로도 시각화

## 폴더 구조

```
sowol_poem_analysis/
├── .gitignore
├── LICENSE
├── requirements.txt
├── preprocess.py        # 시 데이터 로드 및 정제
├── emotion_model.py     # KPoEM 모델 래퍼
├── analyze.py           # 시/행 단위 분석 + 구조화
├── visualize.py         # 히트맵/막대그래프 시각화
├── main.py              # 전체 파이프라인 CLI
├── data/
│   ├── raw/
│   │   └── jindallaekkot_9poems.json   # 시 9편 원문 (퍼블릭 도메인)
│   └── output/                         # 분석 결과 JSON + 그래프 (재생성 가능)
└── README.md
```
