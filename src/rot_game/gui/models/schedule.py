"""Schedule interval data model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScheduleInterval:
    """Represents a single frequency schedule interval.
    
    Attributes:
        frequency_khz: Ultrasound frequency in kHz
        duration_s: Duration of this interval in seconds
        angular_velocity_rad_s: Angular velocity in radians per second
        delay_after_s: Delay after this interval in seconds
    """
    frequency_khz: float
    duration_s: float
    angular_velocity_rad_s: float
    delay_after_s: float = 0.0
