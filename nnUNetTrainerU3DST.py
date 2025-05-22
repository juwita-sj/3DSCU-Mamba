from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.U3DST import get_umamba_bot_from_plans
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
import os

class nnUNetTrainerU3DST(nnUNetTrainer):
    """
    Residual Encoder + UMamba Bottleneck + Residual Decoder + Skip Connections
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_umamba_bot_from_plans(plans_manager, dataset_json, configuration_manager,
                                              num_input_channels, deep_supervision=enable_deep_supervision)
        print("Current working directory:", os.getcwd())
        with open("model_U3DST.txt", "w") as file:
            file.write(str(model))
        print("U3DST: {}".format(model))

        return model
    """
    def train_epoch(self):
        self.network.train()
        total_loss = 0

        # Loop over your data loader
        for i, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass: dapatkan output deep supervision dan adaptive weights
            segmentation_output, adaptive_weights = self.network(data)

            # Inisialisasi total loss untuk current batch
            current_loss = 0

            # Loop over segmentation outputs dan hitung weighted deep supervision loss
            for j, seg_output in enumerate(segmentation_output):
                loss = self.loss_function(seg_output, target)  # Contoh loss function
                current_loss += adaptive_weights[j] * loss  # Menggunakan bobot adaptif

            # Optimizer dan backward propagation
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()

            total_loss += current_loss.item()

        print(f"Train Loss: {total_loss / len(self.train_dataloader)}")
        print("Segmentation output shape:", [seg.shape for seg in segmentation_output])
        print("Adaptive weights:", adaptive_weights)
        """