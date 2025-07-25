import unittest
from src.core.config import Config

class TestCoreConfig(unittest.TestCase):
    def test_config_load(self):
        cfg = Config(config_path='test_config.json')
        self.assertIsInstance(cfg.data, dict)
        self.assertIn('paths', cfg.data)
        self.assertIn('models_dir', cfg.data['paths'])
        # Clean up test config file
        import os
        if os.path.exists('test_config.json'):
            os.remove('test_config.json')

if __name__ == '__main__':
    unittest.main() 