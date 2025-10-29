import torch
from models.base.KgeModel import KgeModel


class DistMult(KgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emb_set_node = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty((self.num_set_node, self.emb_size))).to(self.device))

        self.emb_relations = torch.nn.Parameter(torch.nn.init.xavier_normal_(
            torch.empty((self.num_relations, self.emb_size))).to(self.device))

        if len(self.inductive_embs) > 0:
            # if there are inductive embeddings
            # there should be a trainable projection
            # even if the setting is non-inductive
            self.inductive_proj = torch.nn.Linear(self.inductive_embs.shape[-1], self.emb_size).to(torch.float32).to(self.device)
            self.set_trainable_params([self.emb_set_node, self.emb_relations, *list(self.inductive_proj.parameters()), self.inductive_embs])
        else:
            self.set_trainable_params([self.emb_set_node, self.emb_relations])

    def train_step_pre(self):
        pass

    def train_step_post(self):
        pass

    def lookup(self, inputs):
        """
        head = self.emb_set_node[inputs[0]]
        rel = self.emb_relations[inputs[2]]
        tail = self.emb_set_node[inputs[1]]
        return head, tail, rel
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

        # Assign inductive embeddings using index selection
        if len(head_inductive_indices) > 0:
            inductive_head_ids = head_input[head_inductive_indices]
            ind_head_idx = torch.searchsorted(self.inductive_ids, inductive_head_ids)  # Indices in inductive embeddings
            if self.inductive_embs.shape[-1] != self.emb_set_node.shape[-1]:
                head[head_inductive_indices] = self.inductive_proj(self.inductive_embs[ind_head_idx])
            else:
                head[head_inductive_indices] = self.inductive_embs[ind_head_idx]

        if len(tail_inductive_indices) > 0:
            inductive_tail_ids = tail_input[tail_inductive_indices]
            ind_tail_idx = torch.searchsorted(self.inductive_ids, inductive_tail_ids)  # Indices in inductive embeddings
            if self.inductive_embs.shape[-1] != self.emb_set_node.shape[-1]:
                tail[tail_inductive_indices] = self.inductive_proj(self.inductive_embs[ind_tail_idx])
            else:
                tail[tail_inductive_indices] = self.inductive_embs[ind_tail_idx]

        # Assign regular embeddings for head and tail
        if len(head_regular_indices) > 0:
            head[head_regular_indices] = self.emb_set_node[head_input[head_regular_indices]]

        if len(tail_regular_indices) > 0:
            tail[tail_regular_indices] = self.emb_set_node[tail_input[tail_regular_indices]]

        # Lookup relation embeddings as usual
        rel = self.emb_relations[relation_input]

        return head, rel, tail


    def score(self, head, tail, rel) -> float:
        out = torch.sigmoid(torch.sum(head * tail * rel, dim=-1))
        return out

    def loss(self, inputs, labels):
        head, tail, rel = self.lookup(inputs)

        scores = self.score(head, tail, rel)
        batch_loss = torch.nn.functional.binary_cross_entropy(
            scores.to(self.device),
            labels.to(self.device))

        reg_loss = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(rel ** 2)) / 3

        batch_loss += self.reg_term * (1 / 2) * reg_loss

        return batch_loss, scores
