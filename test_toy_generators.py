import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch package is not installed")
class ToyGeneratorTests(unittest.TestCase):
    def test_training_data_is_bundled_handwritten_digits(self):
        from models.toy_generators import handwritten_digits

        images, labels = handwritten_digits(n=100)
        self.assertEqual(tuple(images.shape), (100, 64))
        self.assertGreaterEqual(float(images.min()), 0.0)
        self.assertLessEqual(float(images.max()), 1.0)
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLessEqual(int(labels.max()), 9)

    def test_autoencoder_validates_input_shape(self):
        import torch
        from models.autoencoder import Autoencoder

        model = Autoencoder(input_dim=6, latent_dim=2)
        reconstruction, latent = model(torch.zeros(3, 6))
        self.assertEqual(tuple(reconstruction.shape), (3, 6))
        self.assertEqual(tuple(latent.shape), (3, 2))
        with self.assertRaisesRegex(ValueError, "Expected 6 features"):
            model(torch.zeros(3, 5))

    def test_all_training_demos_return_samples_and_loss(self):
        from models.toy_generators import train_demo

        names = ("Autoencoder", "VAE", "GAN", "DCGAN", "Conditional GAN", "CycleGAN concept")
        for name in names:
            with self.subTest(model=name):
                result = train_demo(name, steps=2)
                self.assertIn("output", result)
                self.assertGreater(len(result["output"]), 0)
                self.assertTrue(float(result["loss"]) >= 0)
                if name != "CycleGAN concept":
                    self.assertEqual(len(next(iter(result["loss_history"].values()))), 2)


if __name__ == "__main__":
    unittest.main()
