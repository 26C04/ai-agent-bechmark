# OCR Benchmark Report

## Run identity

| Field | Value |
| --- | --- |
| Run ID | unit-test |
| Started at (UTC) | 20260724T010203Z |
| Model tag | gemma4:12b |
| Model digest | sha256:synthetic |
| Prompt name | base |
| Prompt hash | abcdef012345 |
| Split | dev |
| Dataset fingerprint | 1111111111111111111111111111111111111111111111111111111111111111 |
| Preprocess version | v1 |
| Ollama version | fake-ollama |
| Seed | 20260721 |
| Temperature | 0 |
| Participation | reference (not eligible for primary comparison) |
| GPU fully loaded | no |
| Operating system | Synthetic OS |
| GPU | N/A |
| Adapter kind | fake (test adapter) |

## Primary metrics

| Metric | Value |
| --- | --- |
| Exact match | 100.0% (1/1; Wilson 95% CI 0.207 to 1.000) |
| Schema valid | 100.0% |
| First-attempt JSON valid | 100.0% |
| Success after retry | N/A |
| Needs review | 0/1 |

## Field accuracy

| Field | Accuracy | 95% CI |
| --- | --- | --- |
| customer_name | 100.0% | N/A |
| order_no | 100.0% | N/A |
| delivery_date | 100.0% | N/A |
| part_no | 100.0% | 1.000 to 1.000 |
| material | 100.0% | 1.000 to 1.000 |
| num_pieces | 100.0% | 1.000 to 1.000 |

## Source-kind breakdown

| Source kind | Documents | Exact match | Schema valid |
| --- | ---: | ---: | ---: |
| scan | 1 | 100.0% | 100.0% |

## Error categories

| Error tag | Documents |
| --- | ---: |
| None | 0 |

## Latency

| Metric | Value |
| --- | ---: |
| Warm p50 | 10 ms |
| Warm p95 | 10 ms |
| Documents above 30,000 ms | 0/1 |
| Mean preprocess | 1 ms |
| Mean load | 2 ms |
| Mean infer | 3 ms |
| Mean parse_validate | 4 ms |
| Mean total | 10 ms |

## Tokens

| Metric | Value |
| --- | ---: |
| Prompt tokens | 10 |
| Output tokens | 5 |
| Mean prompt tokens/document | 10.0 |
| Mean output tokens/document | 5.0 |
| Mean tokens/sec | 2.5 |

This aggregate report intentionally excludes document IDs, predictions, raw model responses, and ground-truth values.
