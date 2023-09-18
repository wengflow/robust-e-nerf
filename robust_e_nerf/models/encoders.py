import math
import torch


class PositionalEncoder(torch.nn.Module):
    """
    Implementation references:
        1. https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L22
        2. https://github.com/lioryariv/idr/blob/main/code/model/embedder.py
        3. https://github.com/bmild/nerf/issues/12
    """

    def __init__(
        self,
        num_pos_encoding_octaves,
        euclidean_space_scale,
        include_input                       # [Reference 3]
    ):
        super().__init__()
        assert isinstance(num_pos_encoding_octaves, int) \
               and num_pos_encoding_octaves > 0
        assert isinstance(euclidean_space_scale, (int, float)) \
               and euclidean_space_scale > 0.
        assert isinstance(include_input, bool)
        self.include_input = include_input

        # deduce the dimensions of a positional-encoded scalar
        self.dims = 2 * num_pos_encoding_octaves
        if include_input:
            self.dims += 1

        # collate the positional encoding angular frequencies
        scale = 1 / euclidean_space_scale   # [Reference 3]
        self.register_buffer(
            "omega",
            2 ** torch.arange(num_pos_encoding_octaves) * math.pi * scale
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        Returns:
            pos_encoded_input (torch.Tensor): Positional-encoded input data
                                              with shape
                                              (*input.shape, self.dims)
        """
        input_shape_len = len(input.shape)
        reshaped_omega = self.omega.view(               # (input_shape_len of 1s, num_pos_encoding_octaves)
            *([1] * input_shape_len), -1
        )
        reshaped_input = input.unsqueeze(-1)            # (*input.shape, 1)
        scaled_input = reshaped_omega * reshaped_input  # (*input.shape, num_pos_encoding_octaves)

        pos_encoded_input = torch.sin(scaled_input)     # (*input.shape, num_pos_encoding_octaves)
        pos_encoded_input = torch.cat(                  # (*input.shape, 2 * num_pos_encoding_octaves)
            (pos_encoded_input, torch.cos(scaled_input)), dim=-1
        )
        if self.include_input:
            pos_encoded_input = torch.cat(              # (*input.shape, 2 * num_pos_encoding_octaves)
            (reshaped_input, pos_encoded_input), dim=-1 # (*input.shape, 2 * num_pos_encoding_octaves + 1 / self.dims)
        )

        return pos_encoded_input
