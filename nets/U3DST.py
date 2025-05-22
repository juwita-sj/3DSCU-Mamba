import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba

#############################################
# Tambahan: Modul 3DS2T untuk memproses fitur volumetrik secara 3D
#############################################
class ThreeDS2T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, stride=1):
        """
        Modul ini menggunakan 3D convolution untuk mengubah output encoder (3D) menjadi token yang
        mengandung informasi lokal secara volumetrik.
        Args:
            in_channels: jumlah channel input (misal, fitur dari encoder).
            out_channels: jumlah channel output (kami pakai sama dengan in_channels agar kompatibel).
            kernel_size: ukuran kernel konvolusi 3D.
            pool_size: ukuran kernel average pooling 3D.
            stride: stride konvolusi.
        """
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size//2)
        self.relu = nn.ReLU()
        #self.pool = nn.AvgPool3d(kernel_size=pool_size)
    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.conv(x)
        x = self.relu(x)
        #x = self.pool(x)
        return x

#############################################
# Kelas UNetResDecoder (tidak diubah)
#############################################
class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have resolution stages - 1 (n_stages in encoder - 1), here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=encoder.conv_bias,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        assert len(skip_sizes) == len(self.stages)
        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

#############################################
# Kelas UMambaBot: modifikasi hanya di bagian integrasi 3DS2T
#############################################
class U3DST(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have resolution stages. here: {n_stages}. n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries as we have resolution stages. here: {n_stages} stages, so it should have {n_stages - 1} entries. n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        # layer norm
        self.ln = nn.LayerNorm(features_per_stage[-1])
        # Sisipkan modul 3DS2T di sini; output dari encoder akan diproses dengan 3DS2T terlebih dahulu
        self.threeDS2T = ThreeDS2T(in_channels=features_per_stage[-1], out_channels=features_per_stage[-1],
                                    kernel_size=3, pool_size=2, stride=1)


        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        middle_feature = skips[-1]
        middle_feature = self.threeDS2T(middle_feature)  # ThreeDS2T diproses di sini
        skips[-1] = middle_feature
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or batch channel. Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_bot_from_plans(plans_manager: PlansManager,
                              dataset_json: dict,
                              configuration_manager: ConfigurationManager,
                              num_input_channels: int,
                              deep_supervision: bool = True):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)
    segmentation_network_class_name = 'U3DST'
    network_class = U3DST
    kwargs = {
        'U3DST': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    return model
