from typing import Dict, List, Tuple, Union
import numpy as np
from pytorch_forecasting import create_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric as LightningMetric
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.metrics import MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models.nn import LSTM
from utils import get_embedding_size
from metrics import MAE

# ctrl + shift + . to see all classes


class MyScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MyInterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_head = n_head
        self.d_model = d_model

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList(nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head))
        self.k_layers = nn.ModuleList(nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head))
        self.attention = MyScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class MyTimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable

        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0

        return upsampled

    def forward(self, x: torch.Tensor):
        if len(x.size()) <= 2:  # This case assumes x does not have a time dimension.
            return self.interpolate(x)

        # squeeze timesteps and samples into a single axis

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)
        # reshape y to match

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples,output_size)

        return y


class MyGatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.hidden_size = hidden_size or input_size

        # glu splits the input in half and applies a sigmoid to one half and a linear to the other
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        # this returns the tensor half the size of the input in the last dim, thats why we multiply the output layer
        # by 2 in the linear layer
        x = F.glu(x, dim=-1)
        return x


class MyAddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            # if input size is not the same as skip size, we need to resample(resize, linearly 
            # interpolate) the skip connection
            self.resample = MyTimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()

        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)

        return output


class MyGatedAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = MyGatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = MyAddNorm(
            self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add
        )  # this is passed not as a self. because it is not used as a parameter of this class only to initialize other things

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)

        return output


class MyResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int, trainable_add: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size

        if self.input_size != self.output_size:
            # if the input size is not the same as the output size, we need to resize the output
            self.resample = MyTimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()

        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2  # does this come from the paper?

            output = self.norm(x)

        return output


class MyGatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resampled_norm = MyResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)

        self.elu = nn.ELU() # why is this elu and not relu? 

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

        self.gate_norm = MyGatedAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_normal_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resampled_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)

        return x


class MyVariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, MyGatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
    ):
        """
        Calculate weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = MyGatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = MyGatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = MyResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = MyGatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            ## get all of the embeddings from all of the variables and just combine them, very simple
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            #
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)

        else:  # for one input, do not perform variable selection, just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  # batch_size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)
            else:  # ndim == 2 -> batch_size, time, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)

        # the outputs are a weighted sum of the importance of each variable for the current time step?
        return outputs, sparse_weights


class MyTimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)
        x_reshape = x.contiguous().view(-1, x.size(1))

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class MyMultiEmbedding(nn.Module):
    """
    Embedding layer for categorical variables including groups of categorical variables.

    Possibly could be the static covariate encoder layer, not sure

    """

    def __init__(
        self,
        embedding_sizes: Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]],
        x_categoricals: List[str] = None,
        categorical_groups: Dict[str, List[str]] = {},
        embedding_paddings: List[str] = [],
        max_embedding_size: int = None,
    ):
        super().__init__()

        if isinstance(embedding_sizes, dict):
            self.concat_output = False

            # input data checks
            assert x_categoricals is not None, "x_categoricals must be provided."  # groups(fueltype)
            categorical_group_variables = [name for names in categorical_groups.values() for name in names]
            if len(categorical_groups) > 0:
                assert all(
                    name in embedding_sizes for name in categorical_groups
                ), "categorical_groups must be in embedding_sizes."
                assert not any(
                    name in embedding_sizes for name in categorical_group_variables
                ), "group variable in categorical_group must not be in embedding_sizes"
                assert all(
                    name in x_categoricals for name in categorical_group_variables
                ), "group variables in categorical_groups must be in x_categoricals."
            assert all(
                name in embedding_sizes for name in embedding_sizes if name not in categorical_group_variables
            ), (
                "all variables in embedding_sizes must be in x_categoricals - but only if"
                "not already in categorical_groups."
            )
        else:
            assert (
                x_categoricals is None and len(categorical_groups) == 0
            ), " If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty."
            # number embeddings based on order
            embedding_sizes = {str(name): size for name, size in enumerate(embedding_sizes)}
            x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True

        # infer embedding size if not determined
        self.embedding_sizes = {
            name: (size, get_embedding_size(size)) if isinstance(size, int) else size
            for name, size in embedding_sizes.items()
        }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals

        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size

            if name in self.categorical_groups:  # embedding bag if related embedding
                self.embeddings[name] = MyTimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0], embedding_size, padding_idx=padding_idx
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum(s[1] for s in self.embedding_sizes.values())
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    @property
    def output_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_vectors = {}

        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    x[
                        ...,  # select all preceding dimensions
                        [self.x_categoricals.index(cat_name) for cat_name in self.categorical_groups[name]],
                    ]
                )
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])
        if self.concat_output:
            return torch.cat(list(input_vectors.values()), dim=-1)
        else:
            return input_vectors


class MyTemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,  # must pass this up to BaseModelWithCovariates -> BaseModel to override default
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,  # same here as loss
        **kwargs,
    ):
        # these have to be here to cover the mutable default arguments pitfall, need to unsure each instance gets its own instance
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = QuantileLoss()

        # this comes from Superclass of LightningModule -> HyperparametersMixIn, saves params in hparams attribute
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])

        assert isinstance(loss, LightningMetric), "Loss must be Pytorch Lightning 'Metric'"
        # override defaults in BaseModel
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # start to process inputs gl

        # this gets the embedding tensors, not sure what the shape is yet
        self.input_embeddings = MyMultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals  # from base model -> """List of all continuous variables in model"""
            }  # static_categoricals + time_varying_categorical_encoder + time_varying_categorical_decoder,
        )  # {name:Linear layer(1,10)} for all continuous vars. hidden_continuous_sizes can be a dict of different sizes

        # variable selection

        ## variable selection for static variables
        static_input_sizes = {  # not defined as an attribute just a local variable to be used shortly
            name: self.input_embeddings.output_size[name] for name in self.hparams.static_categoricals
        }  # {'fueltype':5} for my example

        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )  # {'encoder_length':10, 'value_center':10,'value_scale':10}

        self.static_variable_selection = MyVariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.static_categoricals},
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        }

        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = MyGatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = MyGatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = MyVariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {} if not self.hparams.share_single_variable_networks else self.shared_single_variable_grns
            ),
        )
        self.decoder_variable_selection = MyVariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {} if not self.hparams.share_single_variable_networks else self.shared_single_variable_grns
            ),
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = MyGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for hidden state of lstm
        self.static_context_initial_hidden_lstm = MyGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for cell state of lstm
        self.static_context_initial_cell_lstm = MyGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for post lstm static enrichment

        self.static_context_enrichment = MyGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # lstm encoder(history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm

        self.post_lstm_gate_encoder = MyGatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder

        self.post_lstm_add_norm_encoder = MyAddNorm(self.hparams.hidden_size, trainable_add=False)

        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past lstm
        self.static_enrichment = MyGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )
        # attention for long range processing

        self.multihead_attn = MyInterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = MyGatedAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = MyGatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = MyGatedAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)

        if self.n_targets > 1:
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, output_size) for output_size in self.hparams.output_size]
            )
        else:
            self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    def expand_static_context(self, context, timesteps):
        """
        add dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor):
        """
        returns causal mask to apply for self-attention layer
        """

        decoder_length = decoder_lengths.max()

        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
        else:
            """
            there is value in attending to future forecasts if they are made with knowledge currently available
            one possibility is here to use a second attention layer for future attention (assuming different
            effects matter in the future than the past)
            or alternatively using the same layer but allowing forward attention - only masking out non-available data
            and self
            """
            decoder_mask = create_mask(decoder_length, decoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)

        # combine masks along attended time - encoder then decoder

        mask = torch.cat((encoder_mask, decoder_mask), dim=2)

        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimension: n_samples x time x variables
        """

        ######### raw inputs
        ######### static metadata, past inputs, and known future inputs

        # all are 672, config['max_encoder_len'].
        encoder_lengths = x["encoder_lengths"]

        # all are 168, config['max_prediction_len']
        decoder_lengths = x["decoder_lengths"]

        # these are the groups for each time step in the batch. encoder and decoder
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # cat along time dim
        # these are the raw values from the dataframe for the encoder and decoder
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        # num timesteps for each sample in batch
        timesteps = x_cont.size(1)  # encode + decode length
        # this is same as config['max_encoder_len'] probably here as good practice keep references local
        max_encoder_length = int(encoder_lengths.max())  # 672

        # this takes the x_cat (group for each time step encoder and decoder)
        # in shape is [128,840,1]
        input_vectors = self.input_embeddings(x_cat)
        # out shape is [128, 840,5]
        # each of the 5 is the same for each sample because it came from the same number

        # this just puts the raw values from the dataframe into input_vectors. no manipulation, just adds the name
        # and reshapes. these ones are [128,840,1]
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        ########## embedding and variable selection
        # static_variables = ['fueltype', 'encoder_length', 'value_center', 'value_scale']
       # print(f"len of static variables: {len(self.static_variables)}")
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        # static variable selection returns a weighted sum of the importance of each static variable

        # this makes static context (128,32) -> (128,840,32) (encoder + decoder length)
        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }

        # this takes all of the embeddings and kind of like attention skip mechanism, does different
        # transformations(linear and non-linear) on each embedding and combines them
        # then concat all of the embeddings together

        # shape before 32 dicts of (128, 672, 1) -> shape after (128, 672, 32) ?????
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder, static_context_variable_selection[:, :max_encoder_length]
        )
        # same for decoder but only for the decoder variables
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables
        }

        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder, static_context_variable_selection[:, max_encoder_length:]
        )

       # print(f"embeddings_varying_encoder shape: {embeddings_varying_encoder.shape}")
      #  print(f"embeddings_varying_decoder shape: {embeddings_varying_decoder.shape}")
       # print(f"static_embedding shape: {static_embedding.shape}")
        # at this point we have static_embedding, embeddings_varying_encoder and embeddings_varying_decoder

        # LSTM
        # calculate initial size
        # this is shape (128, 32) -> (lstm_layers(4), 128, 32)
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)
        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )
        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder, (hidden, cell), lengths=decoder_lengths, enforce_sorted=False
        )
        # skip connection over lstm

        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_encoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

      #  print(f"lstm_output_encoder shape: {lstm_output_encoder.shape}")
      #  print(f"lstm_output_decoder shape: {lstm_output_decoder.shape}")
        # just combine them
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

      #  print(f"lstm_output shape: {lstm_output.shape}")

        # static enrichment

      #  print(static_embedding.shape)
     #   print(static_embedding)
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # attention

        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )

        # skip connection over attention

        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])
        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains)
        # a skip from the variable selection network
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        if self.n_targets > 1:
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        # to_network_output from Base_model -> TupleOutputMixIn
        return self.to_network_output(
            prediction=self.transform_output(
                output, target_scale=x["target_scale"]
            ),  # transform_output from base_model
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            encoder_lengths=encoder_lengths,
            decoder_lengths=decoder_lengths,
        )
