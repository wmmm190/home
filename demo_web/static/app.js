const input = document.getElementById("audioInput");
const runBtn = document.getElementById("runBtn");
const startRecBtn = document.getElementById("startRecBtn");
const stopRecBtn = document.getElementById("stopRecBtn");
const recordingTimeEl = document.getElementById("recordingTime");
const audioPlayer = document.getElementById("audioPlayer");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("resultText");

let mediaRecorder = null;
let mediaStream = null;
let recordChunks = [];
let recordedBlob = null;
let recordTimerId = null;
let recordStartedAt = 0;
let currentPreviewUrl = "";

function formatDuration(ms) {
  const totalSec = Math.floor(ms / 1000);
  const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const s = String(totalSec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function setPreviewSource(fileOrBlob) {
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = "";
  }
  if (!fileOrBlob) {
    audioPlayer.removeAttribute("src");
    audioPlayer.load();
    return;
  }
  currentPreviewUrl = URL.createObjectURL(fileOrBlob);
  audioPlayer.src = currentPreviewUrl;
}

function setBusy(busy) {
  runBtn.disabled = busy;
  startRecBtn.disabled = busy || (mediaRecorder && mediaRecorder.state === "recording");
  stopRecBtn.disabled = busy || !(mediaRecorder && mediaRecorder.state === "recording");
  runBtn.textContent = busy ? "识别中..." : "识别当前音频";
}

function getCurrentAudioSource() {
  const file = input.files && input.files[0];
  if (file) {
    return { blob: file, filename: file.name };
  }
  if (recordedBlob) {
    return { blob: recordedBlob, filename: "record.webm" };
  }
  return null;
}

function stopTimer() {
  if (recordTimerId) {
    clearInterval(recordTimerId);
    recordTimerId = null;
  }
}

function startTimer() {
  recordStartedAt = Date.now();
  recordingTimeEl.textContent = "00:00";
  stopTimer();
  recordTimerId = setInterval(() => {
    recordingTimeEl.textContent = formatDuration(Date.now() - recordStartedAt);
  }, 200);
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusEl.textContent = "当前浏览器不支持麦克风录音。";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordChunks = [];
    recordedBlob = null;

    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      stopTimer();
      recordingTimeEl.textContent = "00:00";

      if (recordChunks.length > 0) {
        recordedBlob = new Blob(recordChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        setPreviewSource(recordedBlob);
        statusEl.textContent = "录音完成，可直接点击“识别当前音频”。";
      } else {
        statusEl.textContent = "录音为空，请重试。";
      }

      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;
      }

      startRecBtn.classList.remove("recording");
      startRecBtn.disabled = false;
      stopRecBtn.disabled = true;
    };

    mediaRecorder.start();
    startTimer();
    startRecBtn.classList.add("recording");
    startRecBtn.disabled = true;
    stopRecBtn.disabled = false;
    input.value = "";
    statusEl.textContent = "录音中...";
  } catch (err) {
    statusEl.textContent = `无法开始录音: ${err.message}`;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

async function runTranscribe() {
  const source = getCurrentAudioSource();
  if (!source) {
    statusEl.textContent = "请先上传音频或先录一段音频。";
    return;
  }

  const form = new FormData();
  form.append("audio", source.blob, source.filename);

  setBusy(true);
  statusEl.textContent = "上传中，正在识别...";
  resultEl.value = "";

  try {
    const resp = await fetch("/api/transcribe", {
      method: "POST",
      body: form,
    });

    // 服务端异常时可能返回非 JSON，保证前端也能显示详细错误
    const raw = await resp.text();
    let data = null;
    try {
      data = JSON.parse(raw);
    } catch {
      data = { ok: false, error: raw || `HTTP ${resp.status}` };
    }

    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    resultEl.value = data.text || "";
    statusEl.textContent = `识别完成，用时 ${data.elapsed_ms} ms。`;
  } catch (err) {
    statusEl.textContent = `识别失败: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

input.addEventListener("change", () => {
  const file = input.files && input.files[0];
  if (file) {
    setPreviewSource(file);
    recordedBlob = null;
    statusEl.textContent = "已选择上传音频，可点击“识别当前音频”。";
  }
});

startRecBtn.addEventListener("click", startRecording);
stopRecBtn.addEventListener("click", stopRecording);
runBtn.addEventListener("click", runTranscribe);
