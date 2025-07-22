import asyncio
import argparse
from src.agents.super_resolution import SuperResolutionAgent
from PIL import Image
import os

MODEL_LIST = [
    "real_esrgan",
    "esrgan",
    "bsrgan",
    "rrdbnet",
    "swinir",
    "restormer",
    "uformer",
    "nafnet",
    "swin2sr",
    "ipt",
]


def get_test_image():
    # Use a small sample image or generate one if not present
    test_img_path = "test_input.jpg"
    if not os.path.exists(test_img_path):
        img = Image.new("RGB", (128, 128), color="gray")
        img.save(test_img_path)
    return test_img_path


async def run_model(model_name, input_path, output_path):
    agent = SuperResolutionAgent()
    await agent._initialize()
    result = await agent._process(
        {"image": input_path, "model": model_name, "output_path": output_path}
    )
    print(f"Model: {model_name}, Status: {result['status']}, Output: {result['output_path']}")


def main():
    parser = argparse.ArgumentParser(description="Test SuperResolutionAgent with all models.")
    parser.add_argument("--model", type=str, default="all", help='Model to run (or "all")')
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_path = args.input or get_test_image()

    if args.model == "all":
        models = MODEL_LIST
    else:
        models = [args.model]

    async def run_all():
        for model in models:
            output_path = os.path.join(args.output_dir, f"output_{model}.jpg")
            try:
                await run_model(model, input_path, output_path)
            except Exception as e:
                print(f"Model {model} failed: {e}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
