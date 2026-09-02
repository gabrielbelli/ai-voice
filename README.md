# ai-voice

Self-hosted speech-to-text and text-to-speech. Five images, one repository,
deployed as a single app.

```text
services/stt        Parakeet TDT 0.6b v3, ONNX          :8000
services/tts        Kokoro-82M, 54 voices               :8001
services/tts-long   Chatterbox, job queue               :8002
services/gateway    one address in front of the three   :8080
packages/common     the shared wire contract
```

Placeholder README; the real one is written once the import is complete.
