from linformer import Linformer
from vit_pytorch.efficient import ViT

from distributed_auto_differentiation.hooks import ModelHook


def build_vit(
    dim=128,
    seq_len=49 + 1,
    depth=12,
    heads=8,
    k=64,
    image_size=224,
    patch_size=32,
    num_classes=2,
    channels=3,
):
    efficient_transformer = Linformer(
        dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=k  # 7x7 patches + 1 cls-token
    )
    model = ViT(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        transformer=efficient_transformer,
        channels=channels,
    )
    return model


if __name__ == "__main__":
    vit = build_vit()
    hook = ModelHook(vit, layer_names=["Linear"])
    print(vit.modules())
