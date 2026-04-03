import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import torch
from scipy.special import log_softmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval_with_lm import (
    CTCBeamDecoder,
    KenLMScorer,
    clean_text,
    get_logits,
    load_model_and_processor,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "dialect_model_best_full"
DEFAULT_LM_ARPA = PROJECT_ROOT / "lm" / "char_5gram.arpa"


def choose_lm_path() -> Path:
    if DEFAULT_LM_ARPA.exists():
        return DEFAULT_LM_ARPA
    raise FileNotFoundError("No KenLM file found under ./lm (char_5gram.arpa)")


def build_decoder(processor, lm_path: Path, beam_width: int, alpha: float, beta: float):
    ext = lm_path.suffix.lower()
    if ext in {".arpa", ".bin"}:
        lm = KenLMScorer(str(lm_path))
    else:
        raise ValueError(f"Unsupported LM format for demo: {lm_path}")

    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get("[PAD]", 0)
    decoder = CTCBeamDecoder(
        vocab=vocab,
        blank_id=blank_id,
        lm=lm,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
    )
    return decoder


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    model_path = Path(os.getenv("DEMO_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    lm_path = Path(os.getenv("DEMO_LM_PATH", str(choose_lm_path())))
    device = os.getenv("DEMO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    beam_width = int(os.getenv("DEMO_BEAM_WIDTH", "20"))
    lm_weight = float(os.getenv("DEMO_LM_WEIGHT", "0.5"))
    word_score = float(os.getenv("DEMO_WORD_SCORE", "0.0"))

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not lm_path.exists():
        raise FileNotFoundError(f"LM path not found: {lm_path}")

    print(f"[demo] loading model from: {model_path}")
    print(f"[demo] loading lm from: {lm_path}")
    print(f"[demo] device={device}, beam_width={beam_width}, alpha={lm_weight}, beta={word_score}")

    model, processor = load_model_and_processor(str(model_path), device=device)
    decoder = build_decoder(processor, lm_path, beam_width, lm_weight, word_score)

    def transcribe_file(audio_path: str) -> str:
        logits = get_logits(model, processor, audio_path, device=device)
        log_probs = log_softmax(logits, axis=-1)
        pred = decoder.decode(log_probs)
        return clean_text(pred)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model_path=str(model_path),
            lm_path=str(lm_path),
            device=device,
            beam_width=beam_width,
            lm_weight=lm_weight,
            word_score=word_score,
        )

    @app.get("/favicon.ico")
    def favicon():
        # 避免浏览器默认请求导致 404 日志干扰
        return "", 204

    @app.post("/api/transcribe")
    def api_transcribe():
        if "audio" not in request.files:
            return jsonify({"ok": False, "error": "No audio file provided. Field name must be 'audio'."}), 400

        f = request.files["audio"]
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "Empty upload."}), 400

        suffix = Path(f.filename).suffix or ".wav"
        tmp_path = None
        t0 = time.time()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                f.save(tmp_path)

            text = transcribe_file(tmp_path)
            elapsed_ms = int((time.time() - t0) * 1000)
            return jsonify({"ok": True, "text": text, "elapsed_ms": elapsed_ms})
        except Exception as e:
            # 服务器日志打印完整栈，便于排查 500 根因
            print("[demo] /api/transcribe failed")
            print(traceback.format_exc())

            msg = str(e)
            if suffix.lower() in {".webm", ".m4a", ".mp4", ".aac", ".ogg"}:
                msg = (
                    f"{msg}. 可能是音频编解码不被当前环境支持（{suffix}）。"
                    "建议改用 WAV，或在环境中安装 ffmpeg 后重试。"
                )

            return jsonify({"ok": False, "error": msg}), 500
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("DEMO_PORT", "7860"))
    app.run(host=host, port=port, debug=False)
