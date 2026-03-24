# MUC Evaluation Metrics

Метрики до и после MUC-интервенции из Table 3 статьи
**"Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations"** (Ji et al., 2025).

**Модель:** Mistral-7B-Instruct-v0.3
**Датасет:** NQ-Open test, 1000 вопросов
**MUC конфиг:** `iti_method=2`, слои `range(15,32)`, `max_alpha=1.0`, промпт `uncertainty`

---

## Запуск

```bash
cd "SAE smoking-room/muc_metrics"
/opt/homebrew/bin/python3.11 compute_metrics.py
# → results.json
```

---

## Результаты

### Полный тест (1000 вопросов)

| Метрика | Before | After | Δ |
|---|---|---|---|
| **Overall Hallucination Rate ↓** | 0.507 | **0.326** | −0.181 |
| Confident Hallucination Rate ↓ | 0.274 | N/A* | — |
| **Correctness Rate ↑** | 0.469 | 0.389 | −0.080 |
| **Refusal Rate** | 0.029 | **0.377** | +0.348 |
| VU/SU Disagreement Rate ↓ | 0.418 | N/A* | — |
| Correlation VU↔SE ↑ | 0.317 | N/A* | — |
| VU for Incorrect ↑ | 0.095 | N/A* | — |
| VU for Correct | 0.029 | N/A* | — |
| Approx SE | 1.485 | 1.650 | +0.165 |

### Только интервенированные вопросы (n=612, alpha > 0)

| Метрика | Before | After | Δ |
|---|---|---|---|
| **Overall Hallucination Rate ↓** | 0.645 | **0.261** | −0.384 |
| Correctness Rate | 0.319 | 0.276 | −0.043 |
| Refusal Rate | 0.042 | **0.611** | +0.569 |

*\*N/A — требует VU after (нужен LLM-judge на ответах после интервенции)*

---

## Интерпретация

**Overall Hallucination Rate** снизился с 0.507 → 0.326 (−36% relative) на полном тесте,
и с 0.645 → 0.261 (−60%) на интервенированных вопросах.
Это хорошо согласуется с результатами статьи (~30% relative reduction, Table 3).

**Refusal Rate** вырос с 0.029 → 0.377 — модель стала значительно чаще отказываться отвечать.
Это ожидаемый trade-off: MUC повышает вербальную неопределённость, некоторые ответы становятся отказами.

**Correctness Rate** снизился с 0.469 → 0.389.
Часть правильных ответов превратилась в отказы (модель стала осторожнее).
Это также ожидаемо (см. сноску 9 в статье).

**SE after** незначительно вырос (1.485 → 1.650) — approximate оценка на основе exact-string dedup
10 сэмплов из MUC (без NLI кластеризации, поэтому значения приблизительные).

---

## Ограничения

1. **Correctness** — substring match (approximate). В статье используется LLM-judge (Appendix A.3).
   Substring match занижает correctness для парафраз ("optic chiasm" vs "the optic chiasma").

2. **VU after** — не вычислен (нет LLM-judge на MUC-ответах).
   Из-за этого недоступны: Confident Hallu Rate, VU/SU Disagreement, Correlation, VU for Incorrect/Correct.

3. **SE after** — approximate (exact-string dedup вместо DeBERTa NLI).

4. **388 вопросов с alpha=0** (detection=0 или SE≈VU) — интервенция не применялась,
   для них используются baseline ответы.

---

## Источники данных

| Файл | Используется для |
|---|---|
| `vuf_checkpoint/.../test.csv` | golden answers, VU before, SE before |
| `vuf_checkpoint/sem_uncertainty/test_0.1.jsonl` | baseline greedy ответы (before) |
| `vuf_checkpoint/sem_uncertainty/test_most_likely_acc.json` | pre-computed accuracy (before) |
| `vuf_checkpoint/verbal_uncertainty/..._vu-llm-judge.json` | VU scores before (per-answer) |
| `vuf_checkpoint/calibration/.../with_vufi_2_...1.0.jsonl` | MUC ответы after (1000 rows) |
| `vuf_checkpoint/detection/.../test_verbal_uncertainty_sentence_semantic_entropy.json` | y_pred детектора |
