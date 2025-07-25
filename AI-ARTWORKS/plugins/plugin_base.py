class PluginBase:
    """
    Base class for all AI-ARTWORK plugins. All plugins must inherit from this class and implement the run() method.
    """
    def __init__(self, context=None):
        self.context = context  # Context or API surface exposed to the plugin

    def run(self, *args, **kwargs):
        """
        Main entry point for the plugin. Must be implemented by all plugins.
        """
        raise NotImplementedError("Plugins must implement the run() method.") 