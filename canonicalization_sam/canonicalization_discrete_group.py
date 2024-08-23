import math
import sys 
sys.path.append("/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/canonicalization_sam")
from typing import Any, Dict, List, Optional, Tuple, Union
import kornia as K
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torchvision import transforms

from can_utils import (
    flip_boxes,
    flip_masks,
    get_action_on_image_features,
    rotate_boxes,
    rotate_masks,
)

class DiscreteGroupImageCanonicalization(torch.nn.Module):
    """
    This class represents a discrete group image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.
    Other discrete group canonicalizers can be derived from this class.

    Methods:
        __init__: Initializes the DiscreteGroupImageCanonicalization instance.
        groupactivations_to_groupelement: Takes the activations for each group element as input and returns the group element.
        get_groupelement: Maps the input image to a group element.
        transformations_before_canonicalization_network_forward: Applies transformations to the input images before passing it through the canonicalization network.
        canonicalize: Canonicalizes the input images.
        invert_canonicalization: Inverts the canonicalization of the output of the canonicalized image.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the DiscreteGroupImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__()
        
        self.canonicalization_network = canonicalization_network
        self.group_type = canonicalization_network.group_type
        self.num_rotations = canonicalization_network.num_rotations
        
        self.gradient_trick = canonicalization_hyperparams.gradient_trick
        self.num_group = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.group_info_dict = {
            "num_rotations": self.num_rotations,
            "num_group": self.num_group,
        }

        self.beta = canonicalization_hyperparams.beta

        assert (
            len(in_shape) == 3
        ), "Input shape should be in the format (channels, height, width)"

        # Define all the image transformations here which are used during canonicalization
        # pad and crop the input image if it is not rotated MNIST
        is_grayscale = in_shape[0] == 1

        self.pad = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Pad(math.ceil(in_shape[-1] * 0.5), padding_mode="edge")
        )
        
        self.crop = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop((in_shape[-2], in_shape[-1]))
        )
        
        self.crop_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop(
                (
                    math.ceil(
                        in_shape[-2] * canonicalization_hyperparams.input_crop_ratio
                    ),
                    math.ceil(
                        in_shape[-1] * canonicalization_hyperparams.input_crop_ratio
                    ),
                )
            )
        )

        self.resize_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Resize(size=canonicalization_hyperparams.resize_shape)
        )

    def groupactivations_to_groupelementonehot(self, group_activations: torch.Tensor) -> torch.Tensor:
        """
        Converts group activations to one-hot encoded group elements in a differentiable manner.

        Args:
            group_activations (torch.Tensor): The activations for each group element.

        Returns:
            torch.Tensor: The one-hot encoding of the group elements.
        """
        group_activations_one_hot = torch.nn.functional.one_hot(
            torch.argmax(group_activations, dim=-1), self.num_group
        ).float()
        group_activations_soft = torch.nn.functional.softmax(
            self.beta * group_activations, dim=-1
        )
        if self.gradient_trick == "straight_through":
            if self.training:
                group_element_onehot = (
                    group_activations_one_hot
                    + group_activations_soft
                    - group_activations_soft.detach()
                )
            else:
                group_element_onehot = group_activations_one_hot
        elif self.gradient_trick == "gumbel_softmax":
            group_element_onehot = torch.nn.functional.gumbel_softmax(
                group_activations, tau=1, hard=True
            )
        else:
            raise ValueError(f"Gradient trick {self.gradient_trick} not implemented")

        # return the group element one hot encoding
        return group_element_onehot
    
    def groupactivations_to_groupelement(self, group_activations: torch.Tensor) -> dict:
        """
        This method takes the activations for each group element as input and returns the group element

        Args:
            group_activations (torch.Tensor): activations for each group element.

        Returns:
            dict: group element.
        """
        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        group_elements_one_hot = self.groupactivations_to_groupelementonehot(
            group_activations
        )

        angles = torch.linspace(0.0, 360.0, self.num_rotations + 1)[
            : self.num_rotations
        ].to(self.device)
        group_elements_rot_comp = (
            torch.cat([angles, angles], dim=0)
            if self.group_type == "roto-reflection"
            else angles
        )

        group_element_dict = {}

        group_element_rot_comp = torch.sum(
            group_elements_one_hot * group_elements_rot_comp, dim=-1
        )
        group_element_dict["rotation"] = group_element_rot_comp

        if self.group_type == "roto-reflection":
            reflect_identifier_vector = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)], dim=0
            ).to(self.device)
            group_element_reflect_comp = torch.sum(
                group_elements_one_hot * reflect_identifier_vector, dim=-1
            )
            group_element_dict["reflection"] = group_element_reflect_comp

        return group_element_dict

    def get_groupelement(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps the input image to a group element.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            dict[str, torch.Tensor]: The corresponding group elements.
        """
        group_activations = self.get_group_activations(x)
        group_element_dict = self.groupactivations_to_groupelement(group_activations)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore
        self.canonicalization_info_dict["group_activations"] = group_activations

        return group_element_dict

    def transformations_before_canonicalization_network_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to the input images before passing it through the canonicalization network.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The pre-canonicalized image.
        """
        x = self.crop_canonization(x)
        x = self.resize_canonization(x)
        return x

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input images.

        Args:
            x (torch.Tensor): The input images.
            targets (Optional[List], optional): The targets for instance segmentation. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized image, and optionally the targets.
        """
        self.device = x.device
        group_element_dict = self.get_groupelement(x)

        x = self.pad(x)

        if "reflection" in group_element_dict.keys():
            reflect_indicator = group_element_dict["reflection"][:, None, None, None]
            x = (1 - reflect_indicator) * x + reflect_indicator * K.geometry.hflip(x)

        x = K.geometry.rotate(x, -group_element_dict["rotation"], padding_mode='border')

        x = self.crop(x)

        if targets:
            # canonicalize the targets (for instance segmentation, masks and boxes)
            image_width = x.shape[-1]

            if "reflection" in group_element_dict.keys():
                # flip masks and boxes
                for t in range(len(targets)):
                    targets[t]["boxes"] = flip_boxes(targets[t]["boxes"], image_width)
                    targets[t]["masks"] = flip_masks(targets[t]["masks"])

            # rotate masks and boxes
            for t in range(len(targets)):
                targets[t]["boxes"] = rotate_boxes(
                    targets[t]["boxes"], group_element_dict["rotation"][t], image_width
                )
                targets[t]["masks"] = rotate_masks(
                    targets[t]["masks"], -group_element_dict["rotation"][t].item()  # type: ignore
                )

            return x, targets

        return x

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization of the output of the canonicalized image.

        Args:
            x_canonicalized_out (torch.Tensor): The output of the canonicalized image.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: The output corresponding to the original image.
        """
        # induced_rep_type = kwargs.get("induced_rep_type", "regular")
        return get_action_on_image_features(
            feature_map=x_canonicalized_out,
            group_info_dict=self.group_info_dict,
            group_element_dict=self.canonicalization_info_dict["group_element"],  # type: ignore
            induced_rep_type="scalar",
        )
    
    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input image.

        This method takes an image as input, applies transformations before forwarding it through the canonicalization network,
        and then returns the group activations.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The group activations.
        """
        x = self.transformations_before_canonicalization_network_forward(x)
        group_activations = self.canonicalization_network(x)
        return group_activations
    
    
    def get_prior_regularization_loss(self) -> torch.Tensor:
        """
        Gets the prior regularization loss.

        Returns:
            torch.Tensor: The prior regularization loss.
        """
        group_activations = self.canonicalization_info_dict["group_activations"]
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(
            self.device
        )
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)


    
    def forward(self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward method for the canonicalization which takes the input data and returns the canonicalized version of the data

        Args:
            x: input data
            targets: (optional) additional targets that need to be canonicalized,
                    such as boxes for promptable instance segmentation
            **kwargs: additional arguments

        Returns:
            canonicalized_x: canonicalized version of the input data

        """
        # call the canonicalize method to obtain canonicalized version of the input data
        return self.canonicalize(x, targets, **kwargs)