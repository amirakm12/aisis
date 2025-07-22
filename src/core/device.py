from typing import Dict, Any


class BaseDeviceAdapter:
    """
    Abstract base class for device adapters (desktop, tablet, mobile, etc.).
    """

    name: str = "base"

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return a dictionary of device capabilities (input, sensors, etc.).
        """
        raise NotImplementedError

    def notify(self, message: str) -> None:
        """Send a notification to the device (if supported)."""
        raise NotImplementedError

    def get_input_type(self) -> str:
        """Return the primary input type (mouse, touch, stylus, etc.)."""
        raise NotImplementedError

    # Add more device-specific methods as needed


class DesktopAdapter(BaseDeviceAdapter):
    name = "desktop"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "input_types": ["mouse", "keyboard"],
            "screen_size": "variable",
            "notifications": True,
        }

    def notify(self, message: str) -> None:
        # TODO: Implement desktop notification
        pass

    def get_input_type(self) -> str:
        return "mouse"


class TabletAdapter(BaseDeviceAdapter):
    name = "tablet"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "input_types": ["touch", "stylus"],
            "screen_size": "medium-large",
            "stylus_pressure": True,
        }

    def notify(self, message: str) -> None:
        # TODO: Implement tablet notification
        pass

    def get_input_type(self) -> str:
        return "touch"


class MobileAdapter(BaseDeviceAdapter):
    name = "mobile"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "input_types": ["touch"],
            "screen_size": "small-medium",
            "camera": True,
        }

    def notify(self, message: str) -> None:
        # TODO: Implement mobile notification
        pass

    def get_input_type(self) -> str:
        return "touch"


# Device adapter registry
DEVICE_ADAPTER_REGISTRY = {
    DesktopAdapter.name: DesktopAdapter(),
    TabletAdapter.name: TabletAdapter(),
    MobileAdapter.name: MobileAdapter(),
}


def detect_device_type() -> str:
    """
    Detect the current device type. (Stub: returns 'desktop' by default.)
    Extend this for platform-specific detection.
    """
    # TODO: Implement real device detection logic
    return "desktop"


def get_current_device_adapter() -> BaseDeviceAdapter:
    """
    Get the adapter for the current device.
    """
    device_type = detect_device_type()
    return DEVICE_ADAPTER_REGISTRY.get(device_type, DesktopAdapter())


"""
How to extend:
- Subclass BaseDeviceAdapter for new device types (e.g., smart TV, AR headset).
- Implement device-specific methods and capabilities.
- Add to DEVICE_ADAPTER_REGISTRY.

This structure supports future expansion for device-specific features and
platforms.
"""
