# NPU Model Checkpoint Adaptations

Adapted `trust_remote_code` files for Ascend 910B (CANN 9.0.0).
Weights unchanged; all modifications GPU-compatible (NPU paths guarded).

Apply to stock checkpoints:
```bash
# Download stock from ModelScope, then:
cp patches/model-checkpoints/vlm/MOSS-VL-Realtime-0708/*.py <vlm_checkpoint>/
cp patches/model-checkpoints/tts/MOSS-TTS-Nano-100M/*.py <tts_checkpoint>/
cp patches/model-checkpoints/tts/MOSS-Audio-Tokenizer-Nano/*.py <codec_checkpoint>/
```

Upstream PRs (recommended long-term):
- https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-0708
- https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M
- https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano

Source of truth: https://github.com/Joiin0392/moss-npu-adaptations
