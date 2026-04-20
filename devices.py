"""
PowerWise AI - Device Simulation Layer
Simulates household electrical devices with ON/OFF control,
energy tracking, and idle detection for auto-shutoff.
"""

from datetime import datetime


class Device:
    def __init__(self, name: str, room: str, watts: int, icon: str = "🔌"):
        self.name = name
        self.room = room
        self.watts = watts
        self.icon = icon

        self.is_on = False
        self.turned_on_at = None
        self.total_seconds_on = 0.0
        self.last_interaction = None
        self.idle_minutes_threshold = 10  # auto-shutoff after X idle minutes

    # ── Power Control ──────────────────────────────────────────
    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            self.turned_on_at = datetime.now()
            self.last_interaction = datetime.now()

    def turn_off(self):
        if self.is_on:
            elapsed = (datetime.now() - self.turned_on_at).total_seconds()
            self.total_seconds_on += elapsed
            self.is_on = False
            self.turned_on_at = None

    def update_interaction(self):
        """Call this when user is actively using the device (resets idle timer)."""
        self.last_interaction = datetime.now()

    # ── Idle Detection ─────────────────────────────────────────
    def get_idle_minutes(self) -> float:
        if not self.is_on or self.last_interaction is None:
            return 0.0
        return (datetime.now() - self.last_interaction).total_seconds() / 60

    def check_auto_shutoff(self) -> bool:
        """Returns True if device has been idle long enough to auto shut off."""
        return self.is_on and self.get_idle_minutes() >= self.idle_minutes_threshold

    # ── Energy Calculation ─────────────────────────────────────
    def get_current_session_kwh(self) -> float:
        """Energy consumed in the current ON session."""
        if not self.is_on or self.turned_on_at is None:
            return 0.0
        elapsed_hours = (datetime.now() - self.turned_on_at).total_seconds() / 3600
        return (self.watts / 1000) * elapsed_hours

    def get_total_kwh(self) -> float:
        """Total energy consumed across all sessions."""
        total_hours = self.total_seconds_on / 3600
        return (self.watts / 1000) * total_hours + self.get_current_session_kwh()

    def get_total_minutes_on(self) -> float:
        total_secs = self.total_seconds_on
        if self.is_on and self.turned_on_at:
            total_secs += (datetime.now() - self.turned_on_at).total_seconds()
        return total_secs / 60

    @property
    def status(self) -> str:
        return "ON" if self.is_on else "OFF"

    def __repr__(self):
        return f"Device({self.name}, {self.watts}W, {self.status})"


def get_default_devices() -> list[Device]:
    """Returns a realistic set of Nigerian household devices."""
    return [
        Device("Living Room Light", "Living Room", 60,  "💡"),
        Device("Bedroom Light",     "Bedroom",     40,  "💡"),
        Device("Kitchen Light",     "Kitchen",     40,  "💡"),
        Device("Living Room Fan",   "Living Room", 75,  "🌀"),
        Device("Bedroom Fan",       "Bedroom",     75,  "🌀"),
        Device("Television",        "Living Room", 120, "📺"),
        Device("Electric Iron",     "Utility",     1000,"🔌"),
        Device("Phone Charger",     "Bedroom",     15,  "📱"),
        Device('Refrigerator', 'Kitchen', 2500, '🧊'),
    ]
