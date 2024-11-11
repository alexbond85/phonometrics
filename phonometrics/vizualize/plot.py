from __future__ import annotations

from typing import Any
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import parselmouth  # type: ignore


class WaveformPlotBuilder:
    def __init__(self, title: str):
        """Initialize with only the title."""
        self.title = title
        self.plot_waveform = False
        self.plot_pitch = False
        self.plot_transcription = False

        # Storage for data relevant to each component
        self.y = None  # Audio waveform
        self.sr = None  # Sample rate
        self.pitch = None  # Pitch data
        self.transcription_data = None

    def with_waveform(self, y: np.ndarray, sr: int) -> WaveformPlotBuilder:
        """Enable waveform plotting and store its parameters."""
        self.plot_waveform = True
        self.y = y  # type: ignore
        self.sr = sr  # type: ignore
        return self

    def with_pitch(self, pitch: parselmouth.Pitch) -> WaveformPlotBuilder:
        """Enable pitch plotting and store pitch data."""
        self.plot_pitch = True
        self.pitch = pitch
        return self

    def with_transcription(
        self, transcription_data: Dict[str, Any]
    ) -> WaveformPlotBuilder:
        """Enable transcription plotting and store transcription data."""
        self.plot_transcription = True
        self.transcription_data = transcription_data  # type: ignore
        return self

    def build(self) -> plt.Figure:
        """Build and return the plot based on selected components."""
        # Reset Matplotlib settings
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "text.usetex": False,
            }
        )

        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot waveform if enabled
        if self.plot_waveform and self.y is not None and self.sr is not None:
            # Prepare time data for waveform
            times = np.linspace(0, len(self.y) / self.sr, len(self.y))
            # Normalize audio waveform
            y_normalized = self.y / np.max(np.abs(self.y))
            ax.plot(
                times, y_normalized, color="gray", alpha=0.5, label="Waveform"
            )

        # Plot pitch if enabled
        if self.plot_pitch and self.pitch is not None:
            # Prepare pitch data
            pitch_values = self.pitch.selected_array["frequency"]
            pitch_values[pitch_values == 0] = np.nan
            ax.scatter(
                self.pitch.xs(), pitch_values, color="r", s=10, label="Pitch"
            )

        # Plot transcription if enabled
        if self.plot_transcription and self.transcription_data:
            transcription = self.transcription_data["transcription"]
            start_times = (
                np.array(self.transcription_data["start_timestamps"]) / 1000
            )  # Convert to seconds
            end_times = (
                np.array(self.transcription_data["end_timestamps"]) / 1000
            )  # Convert to seconds
            probabilities = self.transcription_data["probabilities"]

            # Add transcription rectangles and labels based on probability
            for char, start, end, prob in zip(
                transcription, start_times, end_times, probabilities
            ):
                ax.axvspan(start, end, color=cm.viridis(prob), alpha=0.3)
                mid_time = (start + end) / 2
                ax.text(
                    mid_time,
                    0,
                    char,
                    ha="center",
                    va="center",
                    fontsize=12,
                    alpha=0.9,
                    weight="bold",
                )

        # Set title
        ax.set_title(self.title)
        return fig
