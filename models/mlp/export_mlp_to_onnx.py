import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from mlp_model import SimpleMLP


def main():
    model = SimpleMLP()
    model.eval()

    dummy_input = torch.randn(1, 10, dtype=torch.float32)
    output_path = os.path.join(os.path.dirname(__file__), "simple_mlp.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=12
    )

    print(f"MLP ONNX model exported to: {output_path}")


if __name__ == "__main__":
    main()