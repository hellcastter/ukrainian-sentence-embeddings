from torch import nn
import torch
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(
        self,
        model,
        margin=1.0,
        p=2,
        pool_targets=False,
        use_both_poolings=False,
        loss_type="triplet_loss",
        dynamic=False,
    ):
        super(TripletLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.p = p
        self.pool_targets = pool_targets
        self.loss_fn = None
        self.loss_type = loss_type
        self.dynamic = dynamic

        if self.loss_type == "triplet_loss":
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=self.p)
        elif self.loss_type == "triplet_loss_cosine":
            self.loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=self.margin,
                distance_function=self._cosine_distance,
            )
        else:
            raise Exception(f"{self.loss_type} loss is not supported")

        self.use_both_poolings = use_both_poolings
        self.device = self.model.device

    @staticmethod
    def _cosine_distance(x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return 1 - F.cosine_similarity(x, y)

    def _batch_mean_pooling(
        self,
        token_embeds: torch.Tensor,  # (B, T, D)
        attention_mask: torch.Tensor,  # (B, T)
        target_word_ids=None,  # (B, K) or None
        pool_targets: bool = False,
    ):
        B, T, D = token_embeds.shape
        device = token_embeds.device

        if pool_targets and target_word_ids is not None:
            # target_word_ids: (B, K) with -1 padding
            valid = target_word_ids >= 0  # (B, K)

            # Convert to flat indices in a (B*T) vector
            batch_offsets = (torch.arange(B, device=device) * T).unsqueeze(1)  # (B, 1)
            flat_ids = target_word_ids + batch_offsets  # (B, K)
            flat_ids = flat_ids[valid]  # remove padding

            # Create flat mask and scatter
            flat_mask = torch.zeros(B * T, device=device)
            flat_mask.scatter_(0, flat_ids, 1.0)

            # reshape back to (B, T)
            target_mask = flat_mask.view(B, T)

            mask = target_mask
        else:
            mask = attention_mask.float()

        mask = mask.unsqueeze(-1)  # (B, T, 1)

        pooled = (token_embeds * mask).sum(dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

        return pooled

    @property
    def distance(self):
        if self.loss_type == "triplet_loss":
            return torch.cdist
        elif self.loss_type == "triplet_loss_cosine":
            return self._cosine_distance

    def forward(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        a = self.model(
            batch["anchor_ids"], attention_mask=batch["anchor_mask"]
        ).last_hidden_state
        p = self.model(
            batch["positive_ids"], attention_mask=batch["positive_mask"]
        ).last_hidden_state
        n = self.model(
            batch["negative_ids"], attention_mask=batch["negative_mask"]
        ).last_hidden_state

        if not self.use_both_poolings:
            a = self._batch_mean_pooling(
                a,
                batch["anchor_mask"],
                batch.get("anchor_target_word_ids"),
                pool_targets=self.pool_targets,
            )
            p = self._batch_mean_pooling(
                p,
                batch["positive_mask"],
                batch.get("positive_target_word_ids"),
                pool_targets=self.pool_targets,
            )
            n = self._batch_mean_pooling(
                n,
                batch["negative_mask"],
                batch.get("negative_target_word_ids"),
                pool_targets=self.pool_targets,
            )

            # swap a and p if positive is closer to negative than anchor
            if self.dynamic and self.distance(a, n) > self.distance(p, n):
                a, p = p, n

            return self.loss_fn(a, p, n)

        a1 = self._batch_mean_pooling(a, batch["anchor_mask"], pool_targets=False)
        p1 = self._batch_mean_pooling(p, batch["positive_mask"], pool_targets=False)
        n1 = self._batch_mean_pooling(n, batch["negative_mask"], pool_targets=False)

        a2 = self._batch_mean_pooling(
            a,
            batch["anchor_mask"],
            batch.get("anchor_target_word_ids", None),
            pool_targets=True,
        )
        p2 = self._batch_mean_pooling(
            p,
            batch["positive_mask"],
            batch.get("positive_target_word_ids", None),
            pool_targets=True,
        )
        n2 = self._batch_mean_pooling(
            n,
            batch["negative_mask"],
            batch.get("negative_target_word_ids", None),
            pool_targets=True,
        )

        return (self.loss_fn(a1, p1, n1) + self.loss_fn(a2, p2, n2)) / 2


class MNRLoss(nn.Module):
    def __init__(
        self, model, temperature=0.05, pool_targets=False, use_both_poolings=False
    ):
        super(MNRLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.pool_targets = pool_targets
        self.use_both_poolings = use_both_poolings

        if isinstance(self.model, nn.DataParallel):
            self.device = self.model.module.device
        else:
            self.device = self.model.device

    def _batch_mean_pooling(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_word_ids=None,
        pool_targets: bool = False,
    ):
        B, T, D = token_embeds.shape
        device = token_embeds.device

        if pool_targets and target_word_ids is not None:
            valid = target_word_ids >= 0

            batch_offsets = (torch.arange(B, device=device) * T).unsqueeze(1)
            flat_ids = target_word_ids + batch_offsets
            flat_ids = flat_ids[valid]

            flat_mask = torch.zeros(B * T, device=device)
            flat_mask.scatter_(0, flat_ids, 1.0)

            target_mask = flat_mask.view(B, T)
            mask = target_mask
        else:
            mask = attention_mask.float()

        mask = mask.unsqueeze(-1)

        pooled = (token_embeds * mask).sum(dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

        return pooled

    def _mnrloss(self, a, p):
        """
        Core Multiple Negatives Ranking Loss.
        a: (B, D) anchors
        p: (B, D) positives
        """
        a = F.normalize(a, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)

        logits = torch.matmul(a, p.T) / self.temperature

        labels = torch.arange(a.size(0), device=a.device)

        return F.cross_entropy(logits, labels)

    def forward(self, batch):
        batch = {
            k: v.to(self.device) for k, v in batch.items() if "target_word_ids" not in k
        }

        a = self.model(
            batch["anchor_ids"], attention_mask=batch["anchor_mask"]
        ).last_hidden_state
        p = self.model(
            batch["positive_ids"], attention_mask=batch["positive_mask"]
        ).last_hidden_state

        if not self.use_both_poolings:
            a = self._batch_mean_pooling(
                a,
                batch["anchor_mask"],
                batch.get("anchor_target_word_ids"),
                pool_targets=self.pool_targets,
            )

            p = self._batch_mean_pooling(
                p,
                batch["positive_mask"],
                batch.get("positive_target_word_ids"),
                pool_targets=self.pool_targets,
            )

            return self._mnrloss(a, p)

        # Dual pooling mode – identical logic to your TripletLoss

        a1 = self._batch_mean_pooling(a, batch["anchor_mask"], pool_targets=False)
        p1 = self._batch_mean_pooling(p, batch["positive_mask"], pool_targets=False)

        a2 = self._batch_mean_pooling(
            a,
            batch["anchor_mask"],
            batch.get("anchor_target_word_ids", None),
            pool_targets=True,
        )

        p2 = self._batch_mean_pooling(
            p,
            batch["positive_mask"],
            batch.get("positive_target_word_ids", None),
            pool_targets=True,
        )

        return (self._mnrloss(a1, p1) + self._mnrloss(a2, p2)) / 2


class NTXentLoss(nn.Module):
    def __init__(self, model, temperature=0.05):
        super(NTXentLoss, self).__init__()
        self.model = model  # This is now the ContrastiveModel wrapper
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        if isinstance(self.model, nn.DataParallel):
            self.device = self.model.module.backbone.device
        else:
            self.device = self.model.backbone.device

    def forward(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # The model now returns the PROJECTED and POOLED embeddings directly
        z_i = self.model(batch["view1_ids"], batch["view1_mask"])
        z_j = self.model(batch["view2_ids"], batch["view2_mask"])

        # L2 Norm is still required for Cosine Sim
        z_i = nn.functional.normalize(z_i, p=2, dim=1)
        z_j = nn.functional.normalize(z_j, p=2, dim=1)

        batch_size = z_i.size(0)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = (
            torch.matmul(representations, representations.t()) / self.temperature
        )

        mask = torch.eye(2 * batch_size, device=self.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)

        # Labels and Mask
        labels = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        loss = self.criterion(similarity_matrix, labels)

        return loss / (2 * batch_size)
