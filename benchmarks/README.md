# MCC Benchmark Datasets

This directory contains all benchmark datasets required for evaluating the MCC (Model Confrontation & Collaboration) framework.

## Data Sources

All datasets used in this project were downloaded from their respective official sources and are organized according to the following categories:

### Classical Benchmarks

* **MedQA**: USMLE medical exam questions
  * Source: https://github.com/jind11/MedQA

* **PubMedQA**: Question answering based on PubMed literature abstracts
  * Source: https://pubmedqa.github.io

* **MMLU Medical Subsets**: Massive Multitask Language Understanding - Medical categories
  * Source: https://huggingface.co/collections/openlifescienceai/multimedqa-66098a5b280539974cefe485
  * Includes: Anatomy, Clinical Knowledge, College Biology, College Medicine, Medical Genetics, Professional Medicine

### Next-level Benchmarks: Advanced Reasoning, Metacognition, and Robustness

* **MedXpertQA**: Expert-level medical question answering
  * Source: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA

* **MetaMedQA**: Metacognitive medical question answering dataset
  * Source: https://huggingface.co/datasets/maximegmd/MetaMedQA

* **RABBITS**: Drug name conversion benchmark for testing model robustness
  * Source: https://github.com/BittermanLab/RABBITS

### Long-form Questions

* **MultiMedQA 140**: 140 curated medical questions from multiple sources
  * 20 LiveQA questions: https://github.com/abachaa/LiveQA_MedicalTask_TREC2017
  * 20 MedicationQA questions: https://github.com/abachaa/Medication_QA_MedInfo2019
  * 100 HealthSearchQA questions: Accessed from the study by Natarajan et al. ([Large language models encode clinical knowledge](https://www.nature.com/articles/s41586-023-06291-2))

* **HealthBench**: Comprehensive health question answering evaluation
  * Source: https://github.com/openai/simple-evals

### Diagnostic Dialogue

* **Interactive_OSCE**: Interactive clinical skills examination scenarios from the United Kingdom
  * Source: https://www.thefederation.uk/sites/default/files/documents/Station%202%20Scenario%20Pack%20(16).pdf

## Dataset Organization

Each subdirectory contains test data files (`.json`, `.jsonl` formats) and ground truth files where applicable.


