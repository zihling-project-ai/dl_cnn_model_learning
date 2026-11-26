from .lenet import build_model as build_lenet
from .alexnet import build_model as build_alexnet
from .alexnet_lrn import build_model as build_alexnet_lrn
from .vgg16 import build_model as build_vgg16
from .vgg16_bn import build_model as build_vgg16_bn
from .googlenet import build_model as build_googlenet
from .inception import build_model as build_inception
from .resnet import build_model as build_model_resnet50
from .mobilenet_v1 import build_model as build_mobilenet_v1
from .efficientnet import build_model as build_effnet


MODEL_FACTORY = {
    "lenet": build_lenet,
    "alexnet": build_alexnet,
    "alexnet_lrn" : build_alexnet_lrn,
    "vgg16": build_vgg16,
    "vgg16_bn": build_vgg16_bn,
    "googlenet": build_googlenet,
    "inception": build_inception,
    "resnet50": build_model_resnet50,
    "mobilenet_v1": build_mobilenet_v1,
    "efficientnet_b0": build_effnet,
}



def get_model(name, num_classes, input_shape, aux_logits=True):
    name = name.lower()

    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model name: {name}")

    # GoogLeNet 需要 aux_logits（格式不同）
    if name == "googlenet":
        return MODEL_FACTORY[name](
            input_shape=input_shape,
            num_classes=num_classes,
            aux_logits=aux_logits
        )

    # 其他模型一律用 input_shape, num_classes
    return MODEL_FACTORY[name](
        input_shape=input_shape,
        num_classes=num_classes
    )
