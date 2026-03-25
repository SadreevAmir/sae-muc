# Hallucination Detection: Probe Experiments

Воспроизведение и расширение экспериментов из статьи
**"Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations"** (Ji et al., 2025).

**Модель:** Mistral-7B-Instruct-v0.3
**Датасет:** NQ-Open, 500 train / 500 test
**Задача:** бинарная классификация — галлюцинация (incorrect) vs. корректный ответ

---

## Структура

```
probes_experiment/
├── README.md                  ← этот файл
├── config.py                  ← пути к данным и константы
├── data_utils.py              ← загрузка hidden states, SE, VU, labels
│
├── exp1_probes.py             ← Эксперимент 1: линейные пробы SE/VU → LR
├── exp2_direct.py             ← Эксперимент 2: LR / PCA+LR на hidden states
├── exp3_vuf_projection.py     ← Эксперимент 3: VUF-проекции + MLP
├── compare_results.py         ← Итоговая таблица всех экспериментов
│
├── models/                    ← сохранённые обученные модели (.pkl, .npy)
└── results/                   ← JSON с результатами + comparison_table.md
```

---

## Быстрый старт

```bash
cd probes_experiment

# Все эксперименты последовательно
python exp1_probes.py
python exp2_direct.py
python exp3_vuf_projection.py

# Итоговая таблица
python compare_results.py
```

Результаты сохраняются в `results/`, модели в `models/`.

---

## Описание экспериментов

### Exp 1: Линейные пробы SE/VU → LR  (`exp1_probes.py`)

**Воспроизводит Table 2 и Table 4 из статьи.**

1. Для каждого слоя (15–31) обучается **Ridge-регрессия** предсказывать SE и VU по hidden state последнего токена вопроса.
2. Предсказанные SE и VU подаются в **Logistic Regression** детектор.

Это "probe-predicted" подход: не нужно генерировать 10 сэмплов и запускать LLM-judge.

Также считается baseline с **реальными** SE и VU из CSV ("calculated" в статье).

**Результат:** AUROC = 0.734 (probe) vs 0.777 (raw) — разрыв ~0.04, как в статье.

---

### Exp 2: Прямой LR / PCA+LR на hidden states (`exp2_direct.py`)

LR классификатор обучается **напрямую** на hidden states без промежуточного предсказания SE/VU.

Варианты:
- **Direct LR**: LR на hidden state одного слоя [4096 dim]
- **PCA + LR**: PCA (n=8/16/32/64/128 компонент) → LR
- **Mean layers + LR**: среднее по слоям → LR
- **All layers concat + PCA + LR**: конкатенация всех слоёв → PCA → LR

**Результат:** Лучший вариант — **PCA(32) + LR на L17** — AUROC = 0.724.
Хуже чем через пробы (0.734): 500 примеров мало для прямого обучения в 4096-мерном пространстве.

---

### Exp 3: VUF-проекции + MLP (`exp3_vuf_projection.py`)

**Идея (мотивирована §3 статьи):** вместо предсказания SE/VU числами — проецируем hidden state на VUF-вектор.

```
score(x, l) = h⁽ˡ⁾(x) · r̂_VU(l)
```

Это **интерпретируемо**: скаляр показывает, насколько активации вопроса совпадают с направлением неопределённости на слое l.

Варианты:
- **A. VUF proj all 17 → LR**: 17 скаляров → LR
- **Per-layer proj → LR**: один скаляр на слой → LR
- **B. proj + residual norm → LR**: проекция + норма ортогональной компоненты → LR
- **C. MLP на VUF-проекциях**: нелинейная обработка 17 скаляров
- **D. MLP на hidden states**: двухслойный MLP с dropout на одном слое

**Результат:**
- MLP на hidden states L17 (скрытый=128): AUROC = **0.709** — чуть лучше прямого LR, но хуже проб
- VUF proj+resid_norm L17: AUROC = 0.700 — интерпретируемый подход без SE/VU
- MLP на VUF-проекциях: AUROC = 0.658 — 17 скаляров недостаточно для MLP с 500 примерами

---

## Итоговые результаты

| Подход | AUROC | ACC | Требует |
|---|---|---|---|
| **Raw SE+VU → LR** | **0.777** | 0.710 | 10 сэмплов + LLM-judge |
| Raw SE only → LR | 0.749 | 0.686 | 10 сэмплов |
| Raw VU only → LR | 0.736 | 0.678 | LLM-judge |
| **Probe SE+VU → LR** | **0.734** | 0.672 | только prefill |
| PCA(32)+LR на hs | 0.724 | 0.674 | только prefill |
| MLP на hs L17 | 0.709 | 0.644 | только prefill |
| VUF proj+resid L17 | 0.700 | 0.664 | только prefill, интерпретируемо |
| Direct LR на hs | 0.693 | 0.660 | только prefill |
| VUF proj L17 | 0.692 | 0.654 | только prefill, интерпретируемо |

### Выводы

1. **Probe SE+VU** — лучший "дешёвый" подход (AUROC 0.734): разрыв с calculated всего 0.04.
2. **PCA+LR** — неожиданно конкурентоспособен (0.724): PCA(32) сжимает 4096 → 32 с минимальными потерями.
3. **MLP** лишь незначительно улучшает прямой LR (0.709 vs 0.693) при тех же 500 примерах.
4. **VUF-проекции** — самый интерпретируемый подход (0.700), не требует отдельного обучения SE/VU проб.
5. Все подходы без sampling заметно уступают raw SE+VU (−0.04..−0.08 AUROC).

---

---

### SAE Feature Analysis (`sae_feature_analysis.py`)

Находит **интерпретируемые SAE-фичи**, различающие уверенные и неуверенные ответы модели.

**Что делает:**
1. Загружает предвычисленные активации (certain: 615, uncertain: 35 примеров)
2. Для каждого SAE-слоя (7, 15, 23) прогоняет через SAE encoder (`mistral-7b-res-wg`)
3. Четыре метода ранжирования фич:
   - **Mean diff**: средняя активация uncertain − certain
   - **Frequency diff**: как часто фича ненулевая в каждой группе
   - **Welch's t-test**: статистически значимые различия
   - **Hedge projection**: проекция VUF-вектора на SAE decoder (как в build_intervention_config)
4. Анализирует пересечение top-фич между методами
5. (Опционально) загружает интерпретации с Neuronpedia

**Запуск:**
```bash
# Базовый (CPU)
python probes_experiment/sae_feature_analysis.py

# С GPU
python probes_experiment/sae_feature_analysis.py --device cuda

# С интерпретациями Neuronpedia
python probes_experiment/sae_feature_analysis.py --fetch_neuronpedia

# Больше фич
python probes_experiment/sae_feature_analysis.py --top_k 100
```

**Результат:** `results/sae_feature_analysis.json`

---

## Зависимости

```
torch>=2.0
numpy>=1.26
scikit-learn>=1.3
pandas
joblib
sae-lens>=6.0    # для SAE Feature Analysis
requests         # для Neuronpedia API
```

Установка: `pip install scikit-learn joblib pandas sae-lens requests`

Python: **3.11+**
