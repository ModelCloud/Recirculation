# Local model paths

The following full-precision instruction checkpoints are available to Recirculation on this host:

| Model | Local path | Hugging Face architecture | Decoder layers |
|---|---|---|---:|
| Llama 3.2 1B Instruct | `/monster/data/model/Llama-3.2-1B-Instruct` | `LlamaForCausalLM` | 16 |
| Qwen3 8B | `/monster/data/model/Qwen3-8B` | `Qwen3ForCausalLM` | 36 |
| Gemma 3 1B Instruct | `/monster/data/model/gemma-3-1b-it` | `Gemma3ForCausalLM` | 26 |

These paths were validated with local-only Hugging Face configuration and tokenizer loading, complete expected
Safetensors files, and readable Safetensors metadata on 2026-08-21.

The `/monster/data` directory is owned by `root` and currently has mode `000`. These paths are accessible to the
root-owned Recirculation process, but an unprivileged process requires an appropriate ACL or permission change on the
parent directory.
