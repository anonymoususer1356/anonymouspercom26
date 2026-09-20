from pathlib import Path

import numpy as np


# Normalize an embedding so cosine comparisons are simple dot products.
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


class SpeakerEncoder:
    # Load a CPU ONNX speaker encoder. CAM++ receives log-mel features; ReDimNet receives waveform.
    def __init__(
        self,
        model_path: Path,
        cpu_threads: int | None = None,
        onnx_optimization: str = "all",
        memory_pattern: bool = False,
        allow_spinning: bool = False,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if cpu_threads is not None:
            options.intra_op_num_threads = cpu_threads
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
        if onnx_optimization == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif onnx_optimization == "all":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            raise ValueError(
                "onnx_optimization must be 'disabled' or 'all', got "
                f"{onnx_optimization!r}"
            )
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_rank = len(input_info.shape)
        self.waveform_samples = (
            int(input_info.shape[-1])
            if self.input_rank == 2 and isinstance(input_info.shape[-1], int)
            else 80000
        )
        self.embedding_dimension = (
            int(self.session.get_outputs()[0].shape[-1])
            if isinstance(self.session.get_outputs()[0].shape[-1], int)
            else 192
        )
        self.cpu_threads = cpu_threads

    # Convert 16 kHz mono audio into the 80-bin fbank features required by CAM++.
    def features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        import kaldi_native_fbank as knf

        if sample_rate != 16000:
            raise ValueError("CAM++ expects 16 kHz audio")
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = sample_rate
        options.frame_opts.dither = 0.0
        options.mel_opts.num_bins = 80
        extractor = knf.OnlineFbank(options)
        extractor.accept_waveform(sample_rate, audio.astype(np.float32) * 32768.0)
        extractor.input_finished()
        if extractor.num_frames_ready == 0:
            return np.empty((0, 80), dtype=np.float32)
        frames = np.stack(
            [extractor.get_frame(index) for index in range(extractor.num_frames_ready)]
        ).astype(np.float32)
        return frames - frames.mean(axis=0, keepdims=True)

    # Compute one normalized embedding using the input representation exported by the model.
    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.input_rank == 2:
            if len(audio) >= self.waveform_samples:
                waveform = audio[:self.waveform_samples]
            else:
                waveform = np.pad(audio, (0, self.waveform_samples - len(audio)))
            embedding = self.session.run(None, {self.input_name: waveform[np.newaxis, :]})[0][0]
            return normalize_embedding(embedding)
        frames = self.features(np.asarray(audio, dtype=np.float32))
        if len(frames) == 0:
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        embedding = self.session.run(
            None, {self.input_name: frames[np.newaxis, :, :]}
        )[0][0]
        return normalize_embedding(embedding)

    # Batch equal-shaped ONNX inputs; fixed-window waveform models batch every input together.
    def embed_many(self, audio_items: list[np.ndarray]) -> list[np.ndarray]:
        if self.input_rank == 2:
            if not audio_items:
                return []
            waveforms = []
            for audio in audio_items:
                waveform = np.asarray(audio, dtype=np.float32).reshape(-1)[:self.waveform_samples]
                waveforms.append(np.pad(waveform, (0, max(0, self.waveform_samples - len(waveform)))))
            batch = np.stack(waveforms)
            try:
                embeddings = self.session.run(None, {self.input_name: batch})[0]
                return [normalize_embedding(embedding) for embedding in embeddings]
            except Exception:
                return [self.embed(audio) for audio in audio_items]
        frames_by_index = [self.features(np.asarray(audio, dtype=np.float32)) for audio in audio_items]
        outputs = [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in audio_items]
        groups: dict[int, list[int]] = {}
        for index, frames in enumerate(frames_by_index):
            if len(frames):
                groups.setdefault(len(frames), []).append(index)
        for indices in groups.values():
            batch = np.stack([frames_by_index[index] for index in indices], axis=0)
            embeddings = self.session.run(None, {self.input_name: batch})[0]
            for index, embedding in zip(indices, embeddings):
                outputs[index] = normalize_embedding(embedding)
        return outputs

    # Average embeddings over chunks so long recordings do not exhaust memory.
    def embed_long_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_seconds: float | None = None,
        minimum_seconds: float = 0.5,
    ) -> np.ndarray | None:
        if sample_rate != 16000:
            raise ValueError("Speaker encoders expect 16 kHz audio")
        if chunk_seconds is None:
            chunk_seconds = self.waveform_samples / sample_rate if self.input_rank == 2 else 15.0
        chunk_samples = max(1, int(chunk_seconds * sample_rate))
        minimum_samples = max(1, int(minimum_seconds * sample_rate))
        embeddings = []
        weights = []

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < minimum_samples:
                continue
            embeddings.append(self.embed(chunk))
            weights.append(len(chunk))

        if not embeddings:
            return None

        average = np.average(np.stack(embeddings), axis=0, weights=np.asarray(weights))
        return normalize_embedding(average)

    # Batch chunks from several complete tracks, then reconstruct one fingerprint per track.
    def embed_long_audio_many(
        self,
        audio_items: list[np.ndarray],
        sample_rate: int,
        minimum_seconds: float = 0.5,
    ) -> list[np.ndarray | None]:
        if sample_rate != 16000:
            raise ValueError("Speaker encoders expect 16 kHz audio")
        chunk_samples = self.waveform_samples if self.input_rank == 2 else int(15 * sample_rate)
        minimum_samples = max(1, int(minimum_seconds * sample_rate))
        chunks: list[np.ndarray] = []
        owners: list[tuple[int, int]] = []
        for owner, audio in enumerate(audio_items):
            for start in range(0, len(audio), chunk_samples):
                chunk = np.asarray(audio[start:start + chunk_samples], dtype=np.float32)
                if len(chunk) >= minimum_samples:
                    chunks.append(chunk)
                    owners.append((owner, len(chunk)))
        grouped: list[list[tuple[np.ndarray, int]]] = [[] for _ in audio_items]
        for embedding, (owner, weight) in zip(self.embed_many(chunks), owners):
            grouped[owner].append((embedding, weight))
        results = []
        for entries in grouped:
            if not entries:
                results.append(None)
                continue
            vectors, weights = zip(*entries)
            results.append(normalize_embedding(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights))))
        return results


class CampPlusEncoder(SpeakerEncoder):
    # Compatibility name for existing CAM++ calibration tools and archived runs.
    pass


class ReDimNetEncoder(SpeakerEncoder):
    # ReDimNet ONNX exports accept fixed five-second, 16 kHz waveform tensors.
    pass


# Join selected speech intervals without including the silent full-track gaps.
def concatenate_intervals(
    audio: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    pieces = []
    for start, end in intervals:
        start = max(0, start)
        end = min(len(audio), end)
        if end > start:
            pieces.append(audio[start:end])
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


# Build initial speaker fingerprints from solo speech anywhere in the recording.
def build_solo_profiles(
    mixture: np.ndarray,
    activity: np.ndarray,
    slots: list[int],
    frame_rate: float,
    sample_rate: int,
    encoder: CampPlusEncoder,
    minimum_clip_seconds: float = 0.5,
    maximum_clips: int = 8,
) -> tuple[dict[str, np.ndarray | None], dict[str, dict]]:
    profiles = {}
    details = {}
    selected_activity = activity[:, slots]
    solo_frames = selected_activity.sum(axis=1) == 1

    for slot_index, slot in enumerate(slots):
        speaker = f"SPEAKER_{slot:02d}"
        speaker_frames = solo_frames & selected_activity[:, slot_index]
        intervals = []
        start = None

        for frame in range(len(speaker_frames) + 1):
            is_active = frame < len(speaker_frames) and speaker_frames[frame]
            if is_active and start is None:
                start = frame
            if not is_active and start is not None:
                start_sample = int(start / frame_rate * sample_rate)
                end_sample = int(frame / frame_rate * sample_rate)
                if end_sample - start_sample >= minimum_clip_seconds * sample_rate:
                    intervals.append((start_sample, end_sample))
                start = None

        intervals.sort(key=lambda value: value[1] - value[0], reverse=True)
        intervals = intervals[:maximum_clips]
        profile_audio = concatenate_intervals(mixture, intervals)
        profile = encoder.embed_long_audio(profile_audio, sample_rate)
        profiles[speaker] = profile
        details[speaker] = {
            "source": "solo" if profile is not None else "unavailable",
            "solo_clip_count": len(intervals),
            "solo_seconds": round(len(profile_audio) / sample_rate, 3),
        }

    return profiles, details


# Greedily assign the strongest fingerprint match before considering weaker pairs.
def assign_channels(
    channel_embeddings: list[np.ndarray],
    speakers: list[str],
    profiles: dict[str, np.ndarray | None],
    minimum_similarity: float,
) -> list[dict]:
    remaining_channels = set(range(len(channel_embeddings)))
    remaining_speakers = set(speakers)
    assignments = []

    while remaining_channels and remaining_speakers:
        candidates = []
        for channel in sorted(remaining_channels):
            scores = []
            for speaker in sorted(remaining_speakers):
                profile = profiles.get(speaker)
                if profile is None:
                    continue
                similarity = float(np.dot(channel_embeddings[channel], profile))
                scores.append((speaker, similarity))

            scores.sort(key=lambda item: item[1], reverse=True)
            for rank, (speaker, similarity) in enumerate(scores):
                next_score = (
                    scores[rank + 1][1]
                    if rank + 1 < len(scores)
                    else similarity
                )
                margin = similarity - next_score
                confidence = similarity + 0.5 * margin
                candidates.append((confidence, similarity, margin, channel, speaker))

        if not candidates:
            break

        candidates.sort(reverse=True)
        confidence, similarity, margin, channel, speaker = candidates[0]
        if similarity < minimum_similarity:
            break

        assignments.append(
            {
                "channel": channel,
                "speaker": speaker,
                "method": "fingerprint",
                "similarity": round(similarity, 6),
                "margin": round(margin, 6),
                "confidence": round(confidence, 6),
            }
        )
        remaining_channels.remove(channel)
        remaining_speakers.remove(speaker)

    leftovers_channels = sorted(remaining_channels)
    leftovers_speakers = sorted(remaining_speakers)
    anchored = bool(assignments)

    for channel, speaker in zip(leftovers_channels, leftovers_speakers):
        method = "elimination" if anchored and len(leftovers_channels) == 1 else "ambiguous"
        assignments.append(
            {
                "channel": channel,
                "speaker": speaker,
                "method": method,
                "similarity": None,
                "margin": None,
                "confidence": None,
            }
        )

    return sorted(assignments, key=lambda item: item["channel"])


# Add confidently identified separated speech to a missing speaker profile.
def bootstrap_profiles(
    assignments: list[dict],
    channel_embeddings: list[np.ndarray],
    profiles: dict[str, np.ndarray | None],
    profile_details: dict[str, dict],
) -> None:
    for assignment in assignments:
        if assignment["method"] not in {"fingerprint", "elimination"}:
            continue
        speaker = assignment["speaker"]
        if profiles.get(speaker) is not None:
            continue
        profiles[speaker] = channel_embeddings[assignment["channel"]]
        profile_details[speaker]["source"] = "separated_bootstrap"
