from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class ConvTasNetSeparator:
    # Load the local two- and three-speaker ConvTasNet models on demand.
    def __init__(
        self,
        two_speaker_model: Path,
        three_speaker_model: Path,
        device: str,
        two_speaker_onnx_model: Path | None = None,
        three_speaker_onnx_model: Path | None = None,
        onnx_threads: int = 3,
        onnx_optimization: str = "all",
        onnx_memory_pattern: bool = False,
        onnx_allow_spinning: bool = False,
        input_sample_rate: int = 16000,
        model_sample_rate: int = 8000,
    ):
        self.model_paths = {
            2: two_speaker_model,
            3: three_speaker_model,
        }
        self.models = {}
        self.device = device
        self.onnx_model_paths = {
            2: two_speaker_onnx_model,
            3: three_speaker_onnx_model,
        }
        self.onnx_threads = onnx_threads
        self.torch_threads = onnx_threads
        self.onnx_optimization = onnx_optimization
        self.onnx_memory_pattern = onnx_memory_pattern
        self.onnx_allow_spinning = onnx_allow_spinning
        self.input_sample_rate = input_sample_rate
        self.model_sample_rate = model_sample_rate
        self.onnx_sessions = {}
        self.onnx_unavailable = set()
        self.last_diagnostics = {}

        # The three-speaker fallback is PyTorch, while the two-speaker model is
        # ONNX Runtime. Keep one command-line thread budget meaningful for both.
        # ONNX-only deployments do not need PyTorch installed at run time.
        try:
            import torch
        except ImportError:
            self.torch = None
        else:
            self.torch = torch
            torch.set_num_threads(self.torch_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Another PyTorch component already fixed the inter-op pool size.
                pass

    # Load the matching ONNX graph for CPU separation when one is available.
    def get_onnx_session(self, speaker_count: int):
        model_path = self.onnx_model_paths.get(speaker_count)
        if self.device != "cpu" or model_path is None:
            return None
        if not model_path.exists() or speaker_count in self.onnx_unavailable:
            return None
        if speaker_count in self.onnx_sessions:
            return self.onnx_sessions[speaker_count]

        try:
            import onnxruntime as ort
        except ImportError:
            self.onnx_unavailable.add(speaker_count)
            return None

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.onnx_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.enable_mem_pattern = self.onnx_memory_pattern
        options.add_session_config_entry(
            "session.intra_op.allow_spinning",
            "1" if self.onnx_allow_spinning else "0",
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning",
            "1" if self.onnx_allow_spinning else "0",
        )
        if self.onnx_optimization == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif self.onnx_optimization == "all":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            raise ValueError(
                "onnx_optimization must be 'disabled' or 'all', got "
                f"{self.onnx_optimization!r}"
            )
        self.onnx_sessions[speaker_count] = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return self.onnx_sessions[speaker_count]

    # Return a cached local ConvTasNet model for the requested source count.
    def get_model(self, speaker_count: int):
        from asteroid.models import ConvTasNet
        import torch

        if speaker_count not in self.model_paths:
            raise ValueError(f"No ConvTasNet model for {speaker_count} speakers")
        if speaker_count not in self.models:
            path = self.model_paths[speaker_count]
            if not path.exists():
                raise FileNotFoundError(f"Missing ConvTasNet model: {path}")
            model = ConvTasNet.from_pretrained(str(path))
            self.models[speaker_count] = model.to(self.device).eval()
        return self.models[speaker_count]

    # Separate one overlap region and restore the estimates to mixture scale.
    def separate(
        self,
        mixture: np.ndarray,
        speaker_count: int,
    ) -> list[np.ndarray]:
        input_samples = len(mixture)
        if self.input_sample_rate == self.model_sample_rate:
            model_mixture = mixture.astype(np.float32, copy=False)
        else:
            model_mixture = resample_poly(
                mixture,
                self.model_sample_rate,
                self.input_sample_rate,
            ).astype(np.float32)
        waveform_array = model_mixture[None, :]
        onnx_session = self.get_onnx_session(speaker_count)

        if onnx_session is not None:
            estimates = onnx_session.run(
                ["sources"],
                {"mixture": waveform_array},
            )[0]
            estimates = estimates.squeeze(0)
            backend = "onnxruntime"
        else:
            model = self.get_model(speaker_count)
            import torch
            waveform = torch.from_numpy(waveform_array).to(self.device)
            with torch.no_grad():
                estimates = model.separate(waveform)
            if estimates.dim() == 3:
                estimates = estimates.squeeze(0)
            estimates = estimates.detach().cpu().numpy()
            backend = "pytorch"

        if estimates.ndim != 2 or estimates.shape[0] != speaker_count:
            raise RuntimeError(
                "ConvTasNet returned an unexpected shape: "
                f"{tuple(estimates.shape)} for {speaker_count} speakers"
            )

        channels = []
        for channel in range(speaker_count):
            estimate = estimates[channel]
            if self.input_sample_rate != self.model_sample_rate:
                estimate = resample_poly(
                    estimate,
                    self.input_sample_rate,
                    self.model_sample_rate,
                )
            if len(estimate) < input_samples:
                estimate = np.pad(estimate, (0, input_samples - len(estimate)))
            channels.append(estimate[:input_samples].astype(np.float32))

        stacked = np.stack(channels)
        if not np.all(np.isfinite(stacked)):
            raise RuntimeError("ConvTasNet returned NaN or infinite samples")

        channel_correlation = None
        has_two_non_silent_channels = (
            speaker_count == 2
            and np.std(stacked[0]) > 1e-8
            and np.std(stacked[1]) > 1e-8
        )
        if has_two_non_silent_channels:
            channel_correlation = float(np.corrcoef(stacked[0], stacked[1])[0, 1])

        input_peak = float(np.max(np.abs(mixture))) if len(mixture) else 0.0
        output_peak = float(np.max(np.abs(stacked))) if stacked.size else 0.0
        self.last_diagnostics = {
            "backend": backend,
            "model_sample_rate": self.model_sample_rate,
            "input_peak": round(input_peak, 6),
            "maximum_output_peak": round(output_peak, 6),
            "channel_correlation": (
                round(channel_correlation, 6)
                if channel_correlation is not None
                else None
            ),
        }

        # These models were trained for noisy separation. The residual includes
        # background noise, so forcing the estimated voices to sum exactly to
        # the mixture can create equal-and-opposite channels. Asteroid's
        # separate interface already performs the intended global scaling.
        return [channel for channel in stacked]


class SandglassetSeparator:
    # Keep one fixed-length ONNX session for each supported source count.
    def __init__(
        self,
        two_speaker_onnx_model: Path,
        three_speaker_onnx_model: Path,
        onnx_threads: int = 3,
        input_sample_rate: int = 16000,
        model_sample_rate: int = 16000,
    ):
        self.onnx_model_paths = {
            2: two_speaker_onnx_model,
            3: three_speaker_onnx_model,
        }
        self.onnx_threads = onnx_threads
        self.input_sample_rate = input_sample_rate
        self.model_sample_rate = model_sample_rate
        self.onnx_sessions = {}
        self.model_samples = {}
        self.last_diagnostics = {}

    # Create the same resident ORT session policy used by the existing separator.
    def get_onnx_session(self, speaker_count: int):
        if speaker_count not in self.onnx_model_paths:
            raise ValueError(f"No Sandglasset model for {speaker_count} speakers")
        if speaker_count in self.onnx_sessions:
            return self.onnx_sessions[speaker_count]

        model_path = self.onnx_model_paths[speaker_count]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Sandglasset ONNX model: {model_path}")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.onnx_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 2 or not isinstance(input_shape[1], int):
            raise RuntimeError(
                "Sandglasset ONNX input must have a fixed sample dimension; "
                f"got {input_shape}"
            )
        self.model_samples[speaker_count] = input_shape[1]
        self.onnx_sessions[speaker_count] = session
        return session

    # Run a fixed-length graph while preserving the caller's original duration.
    def separate(self, mixture: np.ndarray, speaker_count: int) -> list[np.ndarray]:
        input_samples = len(mixture)
        if self.input_sample_rate == self.model_sample_rate:
            model_mixture = mixture.astype(np.float32, copy=False)
        else:
            model_mixture = resample_poly(
                mixture,
                self.model_sample_rate,
                self.input_sample_rate,
            ).astype(np.float32)

        session = self.get_onnx_session(speaker_count)
        target_samples = self.model_samples[speaker_count]
        if len(model_mixture) > target_samples:
            raise ValueError(
                f"Sandglasset input is {len(model_mixture)} samples, "
                f"but the fixed graph accepts {target_samples}"
            )
        padded = np.pad(model_mixture, (0, target_samples - len(model_mixture)))
        input_name = session.get_inputs()[0].name
        estimates = session.run(None, {input_name: padded[None, :]})[0]
        estimates = np.asarray(estimates).squeeze(0)
        if estimates.ndim != 2 or estimates.shape[0] != speaker_count:
            raise RuntimeError(
                "Sandglasset returned an unexpected shape: "
                f"{tuple(estimates.shape)} for {speaker_count} speakers"
            )

        channels = []
        for channel in range(speaker_count):
            estimate = estimates[channel]
            estimate = estimate[: len(model_mixture)]
            if self.input_sample_rate != self.model_sample_rate:
                estimate = resample_poly(
                    estimate,
                    self.input_sample_rate,
                    self.model_sample_rate,
                )
            if len(estimate) < input_samples:
                estimate = np.pad(estimate, (0, input_samples - len(estimate)))
            channels.append(estimate[:input_samples].astype(np.float32))

        stacked = np.stack(channels)
        if not np.all(np.isfinite(stacked)):
            raise RuntimeError("Sandglasset returned NaN or infinite samples")
        self.last_diagnostics = {
            "backend": "sandglasset_onnxruntime",
            "model_sample_rate": self.model_sample_rate,
            "fixed_model_samples": target_samples,
            "input_peak": round(float(np.max(np.abs(mixture))), 6) if input_samples else 0.0,
            "maximum_output_peak": round(float(np.max(np.abs(stacked))), 6) if stacked.size else 0.0,
        }
        return [channel for channel in stacked]


# Convert frame-level activity into full-resolution sample masks.
def sample_masks(
    activity: np.ndarray,
    slots: list[int],
    frame_rate: float,
    sample_rate: int,
    sample_count: int,
) -> dict[str, np.ndarray]:
    masks = {}
    for slot in slots:
        speaker = f"SPEAKER_{slot:02d}"
        mask = np.zeros(sample_count, dtype=bool)
        start = None
        for frame in range(activity.shape[0] + 1):
            is_active = frame < activity.shape[0] and activity[frame, slot]
            if is_active and start is None:
                start = frame
            if not is_active and start is not None:
                start_sample = int(start / frame_rate * sample_rate)
                end_sample = min(int(frame / frame_rate * sample_rate), sample_count)
                mask[start_sample:end_sample] = True
                start = None
        masks[speaker] = mask
    return masks


# Fill solo speech directly from the normalized mixture.
def initialize_tracks(
    mixture: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = np.zeros(len(mixture), dtype=np.int16)
    for mask in masks.values():
        count += mask.astype(np.int16)

    solo_masks = {}
    tracks = {}
    for speaker, mask in masks.items():
        solo = mask & (count == 1)
        track = np.zeros_like(mixture)
        track[solo] = mixture[solo]
        solo_masks[speaker] = solo
        tracks[speaker] = track
    return tracks, solo_masks


# Fill active areas that were not source-separated with the original mixture.
def fill_uncovered_activity(
    mixture: np.ndarray,
    tracks: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    solo_masks: dict[str, np.ndarray],
    separated_mask: np.ndarray,
) -> None:
    for speaker in tracks:
        uncovered = masks[speaker] & ~solo_masks[speaker] & ~separated_mask
        tracks[speaker][uncovered] = mixture[uncovered]
