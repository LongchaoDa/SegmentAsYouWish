import escnn
import torch

class ESCNNEquivariantNetwork(torch.nn.Module):
    """
    This class represents an Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant to a discrete group, which can be either rotations or roto-reflections.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        hidden_dim: int, 
        kernel_size: int = 9,
        group_type: str = "rotation",
        group_order: int = 8,
        num_layers: int = 3,
    ):
        """
        Initializes the ESCNNEquivariantNetwork instance.
        Since this is a canonicalization network, we can choose large kernel sizes and small hidden dims. 

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int): The size of the kernel of the convolutional layers.
            group_type (str, optional): It can be either "rotation" (Cn) or "roto-reflection"(Dn). Defaults to "rotation".
            group_order (int, optional): group size (number of rotations). Defaults to 8.
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()

        self.in_channels = inp_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = group_order

        if group_type == "rotation":
            self.gspace = escnn.gspaces.rot2dOnR2(self.num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = escnn.gspaces.flipRot2dOnR2(self.num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            self.num_rotations if group_type == "rotation" else 2 * self.num_rotations
        )

        self.in_type = escnn.nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        self.hid_type = escnn.nn.FieldType(
            self.gspace, [self.gspace.regular_repr] * self.hidden_dim
        )
        self.out_type = escnn.nn.FieldType(
            self.gspace, [self.gspace.regular_repr] * self.out_channels
        )

        modules = [
            escnn.nn.R2Conv(self.in_type, self.out_type, kernel_size, padding = (kernel_size-1)//2),
            # escnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            escnn.nn.ELU(self.out_type, inplace=True),
            escnn.nn.PointwiseMaxPool2D(self.out_type, kernel_size = 2)
        ]
        for _ in range(num_layers - 2):
            modules.append(
                escnn.nn.R2Conv(self.out_type, self.out_type, kernel_size, padding = (kernel_size-1)//2),
            )
            # modules.append(
            #     escnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            # )
            modules.append(
                escnn.nn.ELU(self.out_type, inplace=True),
            )
            modules.append(
                escnn.nn.PointwiseMaxPool2D(self.out_type, kernel_size = 2)
            )

        modules.append(
            escnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),
        )

        self.eqv_network = escnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, num_group_elements).
        """
        x = escnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0],
            self.out_channels,
            self.num_group_elements,
            feature_map.shape[-2],
            feature_map.shape[-1],
        )

        group_activations = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activations