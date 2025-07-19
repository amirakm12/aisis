from .plugin_base import PluginBase


class SamplePlugin(PluginBase):
    """
    Example plugin that prints a message and uses the provided context.
    """

    def run(self, *args, **kwargs):
        print(f"SamplePlugin running! Context: {self.context}")
        return "SamplePlugin executed"
