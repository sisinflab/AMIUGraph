import torch
from torch.nn import functional as F, Parameter
from models.base.KgeModel import KgeModel


class ConvE(KgeModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.first_shape_2d = self.conf.get("first_shape_2d", 16)
        self.bias = self.conf.get("bias", True)
        self.hidden_drop = self.conf.get("hidden_drop", 0.3)
        self.input_drop = self.conf.get("input_drop", 0.2)
        self.feat_drop = self.conf.get("feat_drop", 0.2)
        self.label_smoothing = self.conf.get("label_smoothing", 0.1)
        self.emb_dim1 = self.first_shape_2d
        self.emb_dim2 = self.emb_size // self.first_shape_2d

        if self.emb_dim1 * self.emb_dim2 != self.emb_size:
            raise Exception("Embedding size must be divisible by first_shape_2d")

        if self.emb_dim2 < 3:
            raise Exception("Lower first_shape_2d (kernel size can't be greater than rectangle)")

        self.emb_set_node = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty((self.num_set_node, self.emb_size))).to(self.device))

        self.emb_relations = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty((self.num_relations, self.emb_size))).to(self.device))

        self.inp_drop = torch.nn.Dropout(self.input_drop)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.feat_drop)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.bias, device=self.device)
        self.bn0 = torch.nn.BatchNorm2d(1, device=self.device)
        self.bn1 = torch.nn.BatchNorm2d(32, device=self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_size, device=self.device)

        self.b = torch.nn.Parameter(torch.zeros((self.num_set_node, 1)).to(self.device))

        hidden_size = 32 * (self.emb_dim1 * 2 - 2) * (self.emb_dim2 -2)

        self.fc = torch.nn.Linear(hidden_size, self.emb_size, device=self.device)

        trainable_params = [
            self.emb_set_node,
            self.emb_relations,
            *self.conv1.parameters(),
            *self.bn0.parameters(),
            *self.bn1.parameters(),
            *self.bn2.parameters(),
            self.b,
            *self.fc.parameters()
        ]

        if len(self.inductive_embs) > 0:
            # if there are inductive embeddings
            # there should be a trainable projection
            # even if the setting is non-inductive
            self.inductive_proj = torch.nn.Linear(
                self.inductive_embs.shape[-1],
                self.emb_size
            ).to(torch.float32).to(self.device)
            self.inductive_proj_b = torch.nn.Linear(
                self.inductive_embs.shape[-1],
                1
            ).to(torch.float32).to(self.device)
            trainable_params.extend(list(self.inductive_proj.parameters()))
            trainable_params.extend(list(self.inductive_proj_b.parameters()))
            trainable_params.append(self.inductive_embs)

        self.set_trainable_params(trainable_params, self.conf.get("reg_term", 0.0))

    def train_step_pre(self):
        pass

    def train_step_post(self):
        pass

    def lookup_ent(self, inputs):
        ent_emb = self.emb_set_node[inputs]
        return ent_emb



    def lookup(self, inputs):

        """
        head = self.emb_set_node[inputs[0]]
        rel = self.emb_relations[inputs[2]]
        tail = self.emb_set_node[inputs[1]]
        b = self.b[inputs[1]]
        """
        head_input = torch.tensor(inputs[0].tolist()).to(self.device)
        tail_input = torch.tensor(inputs[1].tolist()).to(self.device)
        relation_input = torch.tensor(inputs[2].tolist()).to(self.device)

        # Create boolean masks for inductive nodes
        head_mask = torch.isin(head_input, self.inductive_ids)
        tail_mask = torch.isin(tail_input, self.inductive_ids)

        # Get the index positions of inductive ids for heads and tails
        head_inductive_indices = torch.nonzero(head_mask, as_tuple=True)[0]
        tail_inductive_indices = torch.nonzero(tail_mask, as_tuple=True)[0]

        # Get the index positions of regular ids for heads and tails
        head_regular_indices = torch.nonzero(~head_mask, as_tuple=True)[0]
        tail_regular_indices = torch.nonzero(~tail_mask, as_tuple=True)[0]

        # Create embeddings for head and tail
        head = torch.empty((len(head_input), self.emb_size), device=self.device)
        tail = torch.empty((len(tail_input), self.emb_size), device=self.device)
        b = torch.empty((len(tail_input), 1), device=self.device)

        # Assign inductive embeddings using index selection
        if len(head_inductive_indices) > 0:
            inductive_head_ids = head_input[head_inductive_indices]
            ind_head_idx = torch.searchsorted(self.inductive_ids,
                                              inductive_head_ids)  # Indices in inductive embeddings
            if self.inductive_embs.shape[-1] != self.emb_set_node.shape[-1]:
                head[head_inductive_indices] = self.inductive_proj(self.inductive_embs[ind_head_idx])
            else:
                head[head_inductive_indices] = self.inductive_embs[ind_head_idx]

        if len(tail_inductive_indices) > 0:
            inductive_tail_ids = tail_input[tail_inductive_indices]
            ind_tail_idx = torch.searchsorted(self.inductive_ids,
                                              inductive_tail_ids)  # Indices in inductive embeddings
            if self.inductive_embs.shape[-1] != self.emb_set_node.shape[-1]:
                tail[tail_inductive_indices] = self.inductive_proj(self.inductive_embs[ind_tail_idx])
            else:
                tail[tail_inductive_indices] = self.inductive_embs[ind_tail_idx]

            b[tail_inductive_indices] = self.inductive_proj_b(self.inductive_embs[ind_tail_idx])

        # Assign regular embeddings for head and tail
        if len(head_regular_indices) > 0:
            head[head_regular_indices] = self.emb_set_node[head_input[head_regular_indices]]

        if len(tail_regular_indices) > 0:
            tail[tail_regular_indices] = self.emb_set_node[tail_input[tail_regular_indices]]
            b[tail_regular_indices] = self.b[tail_input[tail_regular_indices]]

        # Lookup relation embeddings as usual
        rel = self.emb_relations[relation_input]

        return head, tail, rel, b

    def score(self, head, tail, rel, b) -> float:

        # apply padding
        head = head.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel = rel.view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([head, rel], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.sum(x * tail, dim=-1)
        x += b.reshape(x.shape)
        out = torch.sigmoid(x)

        return out

    def loss(self, inputs, labels):
        head, tail, rel, b = self.lookup(inputs)

        scores = self.score(head, tail, rel, b)

        # label smoothing
        labels = labels * (1 - self.label_smoothing) + self.label_smoothing

        batch_loss = torch.nn.functional.binary_cross_entropy(
            scores.to(self.device),
            labels.to(self.device))

        return batch_loss, scores
