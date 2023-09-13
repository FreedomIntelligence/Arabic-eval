from benchmark_eval.evaluation import (
    MMCUEvaluation,
    MMLUEvaluation,
    GSM8KEvaluation,
    CEvalEvaluation,
    MMLUArabicEvaluation,
    EXAMSEvaluation,
    ArabicCultureEvaluation,
)


benchmark2class = {
    'MMCU': MMCUEvaluation,
    'MMLU': MMLUEvaluation,
    'GSM8K': GSM8KEvaluation,
    'CEval': CEvalEvaluation,
    'MMLUArabic': MMLUArabicEvaluation,
    'EXAMS_Arabic': EXAMSEvaluation,
    'ArabicCulture': ArabicCultureEvaluation
}

