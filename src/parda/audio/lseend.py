import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from scipy.ndimage import median_filter


# Turn the public runner setting into the corresponding ONNX Runtime level.
def graph_optimization_level(name: str):
    if name == "disabled":
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if name == "all":
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    raise ValueError(f"onnx_optimization must be 'disabled' or 'all', got {name!r}")


# Load and minimally validate the LS-EEND metadata contract.
def load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "input_dim",
        "state_shapes",
        "conv_delay",
        "real_output_dim",
        "sample_rate",
        "frame_hz",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"LS-EEND metadata is missing: {sorted(missing)}")
    return metadata


class OnlineLSEEND:
    # Keep LS-EEND state across successive audio chunks and timestamp its output.
    def __init__(
        self,
        model_path: Path,
        metadata: dict,
        threads: int,
        onnx_optimization: str = "all",
        memory_pattern: bool = True,
        allow_spinning: bool = False,
        providers: list[str] | None = None,
    ):
        self.metadata = metadata
        self.sample_rate = int(metadata["sample_rate"])
        self.hop_length = int(metadata["hop_length"])
        self.fft_size = int(metadata["n_fft"])
        self.window_length = int(metadata["win_length"])
        self.context = int(metadata["context_recp"])
        self.subsampling = int(metadata["subsampling"])
        self.delay = int(metadata["conv_delay"])
        self.frame_seconds = (
            self.hop_length * self.subsampling / self.sample_rate
        )

        self.window = np.hanning(self.window_length).astype(np.float32)
        self.mel_filter = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.fft_size,
            n_mels=int(metadata["n_mels"]),
        ).astype(np.float32)

        options = ort.SessionOptions()
        options.graph_optimization_level = graph_optimization_level(onnx_optimization)
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.enable_mem_pattern = memory_pattern
        options.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if allow_spinning else "0"
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning", "1" if allow_spinning else "0"
        )
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.state = {
            name: np.zeros(tuple(shape), dtype=np.float32)
            for name, shape in metadata["state_shapes"].items()
        }

        self.audio_buffer = np.zeros(self.fft_size // 2, dtype=np.float32)
        self.feature_sum = np.zeros(int(metadata["n_mels"]), dtype=np.float64)
        self.normalized_frames = []
        self.feature_count = 0
        self.output_count = 0
        self.finished = False

    # Advance the recurrent ONNX state by one feature vector.
    def _step(self, feature: np.ndarray, ingest: float, decode: float) -> dict:
        inputs = {
            "frame": feature.reshape(1, 1, -1).astype(np.float32),
            "enc_ret_kv": self.state["enc_ret_kv"],
            "enc_ret_scale": self.state["enc_ret_scale"],
            "enc_conv_cache": self.state["enc_conv_cache"],
            "dec_ret_kv": self.state["dec_ret_kv"],
            "dec_ret_scale": self.state["dec_ret_scale"],
            "top_buffer": self.state["top_buffer"],
            "ingest": np.array([ingest], dtype=np.float32),
            "decode": np.array([decode], dtype=np.float32),
        }
        values = self.session.run(self.output_names, inputs)
        result = dict(zip(self.output_names, values))
        for name in (
            "enc_ret_kv",
            "enc_ret_scale",
            "enc_conv_cache",
            "dec_ret_kv",
            "dec_ret_scale",
            "top_buffer",
        ):
            self.state[name] = result[name + "_out"]
        return result

    # Materialize ONNX kernels before capture without retaining synthetic state.
    def warmup(self) -> None:
        frame_input = next(item for item in self.session.get_inputs() if item.name == "frame")
        frame_width = int(frame_input.shape[-1])
        self._step(np.zeros(frame_width, dtype=np.float32), 0.0, 0.0)
        self.state = {
            name: np.zeros(tuple(shape), dtype=np.float32)
            for name, shape in self.metadata["state_shapes"].items()
        }

    # Convert one model result into a timestamped real-speaker probability row.
    def _emit(self, result: dict) -> tuple[float, np.ndarray]:
        logits = result["full_logits"].reshape(-1)[1:-1]
        probability = 1.0 / (1.0 + np.exp(-logits))
        timestamp = self.output_count * self.frame_seconds
        self.output_count += 1
        return timestamp, probability.astype(np.float32)

    # Ingest one feature centre after enough right context is available.
    def _ingest_center(self, center: int) -> tuple[float, np.ndarray] | None:
        last_frame = len(self.normalized_frames) - 1
        context_frames = []
        for offset in range(-self.context, self.context + 1):
            source = max(0, min(center + offset, last_frame))
            context_frames.append(self.normalized_frames[source])
        feature = np.concatenate(context_frames)
        should_decode = self.feature_count >= self.delay
        result = self._step(feature, 1.0, 1.0 if should_decode else 0.0)
        self.feature_count += 1
        return self._emit(result) if should_decode else None

    # Accept arbitrary 8 kHz chunks and emit every newly stable decision frame.
    def accept(self, audio_8k: np.ndarray) -> list[tuple[float, np.ndarray]]:
        if self.finished:
            raise RuntimeError("Cannot add audio after OnlineLSEEND.finish()")
        self.audio_buffer = np.concatenate(
            (self.audio_buffer, np.asarray(audio_8k, dtype=np.float32))
        )
        emitted = []

        while len(self.audio_buffer) >= self.fft_size:
            frame = self.audio_buffer[: self.fft_size]
            self.audio_buffer = self.audio_buffer[self.hop_length :]
            spectrum = np.fft.rfft(
                frame[: self.window_length] * self.window,
                n=self.fft_size,
            )
            power = np.abs(spectrum) ** 2
            log_mel = np.log10(
                np.maximum(power @ self.mel_filter.T, 1e-10)
            )
            self.feature_sum += log_mel
            frame_count = len(self.normalized_frames) + 1
            normalized = log_mel - self.feature_sum / frame_count
            self.normalized_frames.append(normalized.astype(np.float32))

            center = len(self.normalized_frames) - 1 - self.context
            if center < 0 or center % self.subsampling != 0:
                continue
            result = self._ingest_center(center)
            if result is not None:
                emitted.append(result)

        return emitted

    # Clamp final right context and drain the model's convolutional delay.
    def finish(self) -> list[tuple[float, np.ndarray]]:
        if self.finished:
            return []
        self.finished = True
        emitted = []

        next_center = self.feature_count * self.subsampling
        while next_center < len(self.normalized_frames):
            result = self._ingest_center(next_center)
            if result is not None:
                emitted.append(result)
            next_center = self.feature_count * self.subsampling

        zero_feature = np.zeros(int(self.metadata["input_dim"]), dtype=np.float32)
        while self.output_count < self.feature_count:
            result = self._step(zero_feature, 0.0, 1.0)
            emitted.append(self._emit(result))

        return emitted


# Extract the exact logmel23 cumulative-mean-normalized LS-EEND frontend.
def extract_features(audio_8k: np.ndarray, metadata: dict) -> np.ndarray:
    hop_length = int(metadata["hop_length"])
    fft_size = int(metadata["n_fft"])
    window_length = int(metadata["win_length"])
    mel_bins = int(metadata["n_mels"])
    context = int(metadata["context_recp"])
    subsampling = int(metadata["subsampling"])
    sample_rate = int(metadata["sample_rate"])

    padding = fft_size // 2
    padded = np.concatenate([np.zeros(padding, dtype=np.float32), audio_8k])
    spectrum = librosa.stft(
        padded,
        n_fft=fft_size,
        win_length=window_length,
        hop_length=hop_length,
        center=False,
    ).T
    mel_filter = librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_size,
        n_mels=mel_bins,
    )
    mel = np.dot(np.abs(spectrum) ** 2, mel_filter.T)
    log_mel = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)

    counts = np.arange(1, len(log_mel) + 1, dtype=np.float64)[:, None]
    cumulative_mean = np.cumsum(log_mel, axis=0, dtype=np.float64) / counts
    normalized = (log_mel - cumulative_mean).astype(np.float32)

    feature_count = len(normalized) // subsampling
    features = []
    for feature_index in range(feature_count):
        center = feature_index * subsampling
        context_frames = []
        for offset in range(-context, context + 1):
            source_index = max(0, min(center + offset, len(normalized) - 1))
            context_frames.append(normalized[source_index])
        features.append(np.concatenate(context_frames))

    return np.asarray(features, dtype=np.float32)


# Run the stateful LS-EEND step model and return real-speaker probabilities.
def run_lseend(
    audio_8k: np.ndarray,
    model_path: Path,
    metadata: dict,
    threads: int,
    onnx_optimization: str = "all",
) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = graph_optimization_level(onnx_optimization)
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    state = {
        name: np.zeros(tuple(shape), dtype=np.float32)
        for name, shape in metadata["state_shapes"].items()
    }
    features = extract_features(audio_8k, metadata)
    convolution_delay = int(metadata["conv_delay"])
    zero_frame = np.zeros((1, 1, int(metadata["input_dim"])), dtype=np.float32)
    logits = []

    def step(frame: np.ndarray, ingest: float, decode: float) -> dict:
        inputs = {
            "frame": frame,
            "enc_ret_kv": state["enc_ret_kv"],
            "enc_ret_scale": state["enc_ret_scale"],
            "enc_conv_cache": state["enc_conv_cache"],
            "dec_ret_kv": state["dec_ret_kv"],
            "dec_ret_scale": state["dec_ret_scale"],
            "top_buffer": state["top_buffer"],
            "ingest": np.array([ingest], dtype=np.float32),
            "decode": np.array([decode], dtype=np.float32),
        }
        values = session.run(output_names, inputs)
        result = dict(zip(output_names, values))
        state["enc_ret_kv"] = result["enc_ret_kv_out"]
        state["enc_ret_scale"] = result["enc_ret_scale_out"]
        state["enc_conv_cache"] = result["enc_conv_cache_out"]
        state["dec_ret_kv"] = result["dec_ret_kv_out"]
        state["dec_ret_scale"] = result["dec_ret_scale_out"]
        state["top_buffer"] = result["top_buffer_out"]
        return result

    for index, feature in enumerate(features):
        should_decode = 1.0 if index >= convolution_delay else 0.0
        result = step(feature.reshape(1, 1, -1), 1.0, should_decode)
        if should_decode:
            logits.append(result["full_logits"].reshape(1, -1))

    missing_outputs = len(features) - len(logits)
    for _ in range(missing_outputs):
        result = step(zero_frame, 0.0, 1.0)
        logits.append(result["full_logits"].reshape(1, -1))

    if not logits:
        return np.zeros((0, int(metadata["real_output_dim"])), dtype=np.float32)

    combined = np.concatenate(logits, axis=0)
    real_logits = combined[:, 1:-1]
    return 1.0 / (1.0 + np.exp(-real_logits))


# Convert probabilities into stable boolean activity tracks per slot.
def activity_matrix(
    probabilities: np.ndarray,
    threshold: float,
    median_width: int = 3,
) -> np.ndarray:
    activity = np.zeros_like(probabilities, dtype=bool)
    for slot in range(probabilities.shape[1]):
        binary = (probabilities[:, slot] >= threshold).astype(np.float32)
        activity[:, slot] = median_filter(binary, size=median_width) >= 0.5
    return activity


# Drop slots that do not contain enough speech to represent a speaker.
def active_slots(
    activity: np.ndarray,
    frame_rate: float,
    minimum_seconds: float,
    forced_speaker_count: int | None,
) -> list[int]:
    durations = activity.sum(axis=0) / frame_rate
    candidates = [
        slot for slot, duration in enumerate(durations) if duration >= minimum_seconds
    ]
    if forced_speaker_count is not None and len(candidates) > forced_speaker_count:
        candidates.sort(key=lambda slot: durations[slot], reverse=True)
        candidates = candidates[:forced_speaker_count]
    return sorted(candidates)


# Turn frame activity into a simple diarisation segment list.
def build_segments(
    activity: np.ndarray,
    slots: list[int],
    frame_rate: float,
    minimum_seconds: float = 0.1,
) -> list[dict]:
    segments = []
    for slot in slots:
        start = None
        for frame in range(activity.shape[0] + 1):
            is_active = frame < activity.shape[0] and activity[frame, slot]
            if is_active and start is None:
                start = frame
            if not is_active and start is not None:
                duration = (frame - start) / frame_rate
                if duration >= minimum_seconds:
                    segments.append(
                        {
                            "start": round(start / frame_rate, 3),
                            "end": round(frame / frame_rate, 3),
                            "speaker": f"SPEAKER_{slot:02d}",
                            "slot": slot,
                        }
                    )
                start = None
    return sorted(segments, key=lambda item: (item["start"], item["speaker"]))


# Group contiguous frames that have the same overlapping speaker set.
def find_overlap_regions(
    activity: np.ndarray,
    slots: list[int],
    frame_rate: float,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict]:
    if not slots:
        return []

    regions = []
    start = None
    current_slots = None

    for frame in range(activity.shape[0] + 1):
        if frame < activity.shape[0]:
            frame_slots = frozenset(slot for slot in slots if activity[frame, slot])
            if len(frame_slots) < 2:
                frame_slots = None
        else:
            frame_slots = None

        if frame_slots != current_slots:
            if current_slots is not None and start is not None:
                end = frame
                duration = (end - start) / frame_rate
                if duration >= minimum_seconds:
                    maximum_frames = max(1, int(maximum_seconds * frame_rate))
                    piece_start = start
                    while piece_start < end:
                        piece_end = min(piece_start + maximum_frames, end)
                        regions.append(
                            {
                                "start": piece_start / frame_rate,
                                "end": piece_end / frame_rate,
                                "slots": sorted(current_slots),
                                "speaker_count": len(current_slots),
                            }
                        )
                        piece_start = piece_end
            start = frame if frame_slots is not None else None
            current_slots = frame_slots

    return regions
