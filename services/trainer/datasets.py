import torch
import ast


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        anchor,
        positive,
        negative,
        tokenizer,
        anchor_target_word_ids=None,
        positive_target_word_ids=None,
        negative_target_word_ids=None,
        seq_len=128,
        pool_targets=False,
    ):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

        self.anchor_target_word_ids = anchor_target_word_ids
        self.positive_target_word_ids = positive_target_word_ids
        self.negative_target_word_ids = negative_target_word_ids

        if self.anchor_target_word_ids is not None:
            self.anchor_target_word_ids = [
                ast.literal_eval(ids) for ids in self.anchor_target_word_ids
            ]

        if self.positive_target_word_ids is not None:
            self.positive_target_word_ids = [
                ast.literal_eval(ids) for ids in self.positive_target_word_ids
            ]

        if self.negative_target_word_ids is not None:
            self.negative_target_word_ids = [
                ast.literal_eval(ids) for ids in self.negative_target_word_ids
            ]

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pool_targets = pool_targets

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        anchor = str(self.anchor[idx])
        positive = str(self.positive[idx])
        negative = str(self.negative[idx])

        tokenized_anchor = self.tokenizer(
            anchor,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        tokenized_positive = self.tokenizer(
            positive,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        tokenized_negative = self.tokenizer(
            negative,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        return_dict = {
            "anchor_ids": torch.tensor(tokenized_anchor["input_ids"], dtype=torch.long),
            "anchor_mask": torch.tensor(
                tokenized_anchor["attention_mask"], dtype=torch.long
            ),
            "positive_ids": torch.tensor(
                tokenized_positive["input_ids"], dtype=torch.long
            ),
            "positive_mask": torch.tensor(
                tokenized_positive["attention_mask"], dtype=torch.long
            ),
            "negative_ids": torch.tensor(
                tokenized_negative["input_ids"], dtype=torch.long
            ),
            "negative_mask": torch.tensor(
                tokenized_negative["attention_mask"], dtype=torch.long
            ),
        }

        if self.pool_targets:
            if self.anchor_target_word_ids is not None:
                return_dict.update(
                    {"anchor_target_word_ids": self.anchor_target_word_ids[idx]}
                )
            else:
                return_dict.update({"anchor_target_word_ids": [None] * self.seq_len})

            if self.positive_target_word_ids is not None:
                return_dict.update(
                    {"positive_target_word_ids": self.positive_target_word_ids[idx]}
                )

            if self.negative_target_word_ids is not None:
                return_dict.update(
                    {"negative_target_word_ids": self.negative_target_word_ids[idx]}
                )

        return return_dict


class PairsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        anchor,
        positive,
        tokenizer,
        anchor_target_word_ids=None,
        seq_len=128,
        pool_targets=False,
    ):
        self.anchor = anchor
        self.positive = positive

        self.anchor_target_word_ids = anchor_target_word_ids

        if self.anchor_target_word_ids is not None:
            self.anchor_target_word_ids = [
                ast.literal_eval(ids) for ids in self.anchor_target_word_ids
            ]

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pool_targets = pool_targets

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        anchor = str(self.anchor[idx])
        positive = str(self.positive[idx])

        tokenized_anchor = self.tokenizer(
            anchor,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        tokenized_positive = self.tokenizer(
            positive,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        return_dict = {
            "anchor_ids": torch.tensor(tokenized_anchor["input_ids"], dtype=torch.long),
            "anchor_mask": torch.tensor(
                tokenized_anchor["attention_mask"], dtype=torch.long
            ),
            "positive_ids": torch.tensor(
                tokenized_positive["input_ids"], dtype=torch.long
            ),
            "positive_mask": torch.tensor(
                tokenized_positive["attention_mask"], dtype=torch.long
            ),
        }

        if self.pool_targets:
            return_dict.update(
                {"anchor_target_word_ids": self.anchor_target_word_ids[idx]}
            )

        return return_dict


class NTXentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        view1,
        view2,
        tokenizer,
        seq_len=128,
        transformation=True,
        mask_prob=0.1,
        shuffle_prob=0.1,
    ):
        assert len(view1) == len(view2), "View1 and View2 must have the same length"

        self.view1 = view1
        self.view2 = view2
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.transformation = transformation

        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob
        self.special_ids = set(self.tokenizer.all_special_ids)

    def __len__(self):
        return len(self.view1)

    def _apply_noise(self, input_ids):
        # 1. Masking
        # Create a mask for content tokens only
        input_ids = torch.tensor(input_ids)
        content_mask = torch.tensor(
            [1 if id not in self.special_ids else 0 for id in input_ids]
        )

        # Random mask logic
        rand = torch.rand(input_ids.shape)
        mask_indices = (rand < self.mask_prob) & (content_mask == 1)
        input_ids[mask_indices] = self.tokenizer.mask_token_id

        # 2. Shuffling (using the logic from Step 2)
        content_indices = torch.where(content_mask == 1)[0]
        if len(content_indices) > 2:
            num_to_shuffle = int(len(content_indices) * self.shuffle_prob)

            if num_to_shuffle > 1:
                idx = torch.randperm(len(content_indices))[:num_to_shuffle]
                shuffled_idx = idx[torch.randperm(len(idx))]

                actual_pos = content_indices[idx]
                shuffled_pos = content_indices[shuffled_idx]
                input_ids[actual_pos] = input_ids[shuffled_pos].clone()

        return input_ids

    def __getitem__(self, idx):
        v1 = str(self.view1[idx])
        v2 = str(self.view2[idx])

        tokenized_v1 = self.tokenizer(
            v1,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        tokenized_v2 = self.tokenizer(
            v2,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        view1_ids = tokenized_v1["input_ids"]
        view1_mask = tokenized_v1["attention_mask"]
        view2_ids = tokenized_v2["input_ids"]
        view2_mask = tokenized_v2["attention_mask"]

        if self.transformation:
            view1_ids = self._apply_noise(view1_ids)
            view2_ids = self._apply_noise(view2_ids)

        return {
            "view1_ids": view1_ids,
            "view1_mask": torch.tensor(view1_mask, dtype=torch.long),
            "view2_ids": view2_ids,
            "view2_mask": torch.tensor(view2_mask, dtype=torch.long),
        }
