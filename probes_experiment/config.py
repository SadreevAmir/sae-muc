"""Shared paths and constants for all experiments."""
import os

# Root of the SAE smoking-room repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_CSV      = f"{ROOT}/datasets__nq_open/Mistral-7B-Instruct-v0.3/train.csv"
TEST_CSV       = f"{ROOT}/datasets__nq_open/Mistral-7B-Instruct-v0.3/test.csv"
TRAIN_PT_DIR   = f"{ROOT}/datasets__nq_open/Mistral-7B-Instruct-v0.3/pipeline_used_layers_last/train"
TEST_PT_DIR    = f"{ROOT}/datasets__nq_open/Mistral-7B-Instruct-v0.3/pipeline_used_layers_last/test"
TRAIN_VU_JUDGE = f"{ROOT}/verbal_uncertainty__outputs/Mistral-7B-Instruct-v0.3_nq_open_train_uncertainty_1.0_vu-llm-judge.json"
TEST_VU_JUDGE  = f"{ROOT}/verbal_uncertainty__outputs/Mistral-7B-Instruct-v0.3_nq_open_test_uncertainty_1.0_vu-llm-judge.json"
TRAIN_SE_PKL   = f"{ROOT}/nq_open/sentence/Mistral-7B-Instruct-v0.3/train_semantic_entropy.pkl"
TEST_SE_PKL    = f"{ROOT}/nq_open/sentence/Mistral-7B-Instruct-v0.3/test_semantic_entropy.pkl"
TRAIN_ACC_JSON = f"{ROOT}/nq_open/sentence/Mistral-7B-Instruct-v0.3/train_most_likely_acc.json"
TEST_ACC_JSON  = f"{ROOT}/nq_open/sentence/Mistral-7B-Instruct-v0.3/test_most_likely_acc.json"

# VUF vectors (uncertainty prompt), shape [32, 4096]
VUF_TRAIN = f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/train/Hs_hedge_universal.pt"
VUF_TEST  = f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/test/Hs_hedge_universal.pt"
VUF_MERGED = f"{ROOT}/calibration__outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt"

LAYERS = list(range(15, 32))   # 17 layers stored in pipeline_used_layers_last
N_TRAIN = 500
N_TEST  = 500

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
