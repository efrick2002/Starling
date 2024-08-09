import json
import os
import uuid
from time import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.p3o_types import P3ORLBatch, P3ORLElement
from trlx.models.modeling_p3o import (
    # MistralModelWithHydraHead,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.p3o_pipeline import P3ORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateP3OTrainer(AccelerateRLTrainer):
    """P3O Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """P3O Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = P3ORolloutStorage(self.tokenizer.pad_token_id, self.tokenizer.padding_side)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(self.model, self.opt, self.scheduler, rollout_loader)

        self.store.clear_history()  # Clear the rollout store

        # Set up a reference model when hydra heads are not used
        if not hasattr(self.model, "frozen_head") and not self.model.peft_type:
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        generate_kwargs = dict(
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            synced_gpus=os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3",
        )
        self.generate_kwargs = {**generate_kwargs, **config.method.gen_kwargs}

        if config.method.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = {**generate_kwargs, **config.method.gen_experience_kwargs}
        else:
            self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        # if config.model.model_path == "openchat/openchat_3.5":
        #     print("Using P3O, return Model-wrapper MistralModelWithHydraHead")
        #     base_model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")
        #     return MistralModelWithHydraHead(base_model=base_model, num_layers_unfrozen=config.model.num_layers_unfrozen)

        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            num_value_layers_unfrozen=config.method.num_value_layers_unfrozen,
            peft_config=self.config.model.peft_config,
        )

    def loss(self, batch: P3ORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logratios = batch.logratios.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        rewards = batch.scalar_rewards.to(self.accelerator.device)
        num_responses_per_query = self.config.method.num_responses_per_query
        # print("num_responses_per_query:", num_responses_per_query)

        # if self.config.model.model_arch_type == "seq2seq":
        #     input_ids = query_tensors
        #     decoder_input_ids = response_tensors
        #     attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
        #     decoder_attention_mask = (
        #         decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
        #     )
        #     decoder_attention_mask[:, 0] = 1

        #     # Forward pass
        #     outputs = self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         decoder_input_ids=decoder_input_ids,
        #         decoder_attention_mask=decoder_attention_mask,
        #     )

        #     logits = outputs.logits
        #     values_pred = outputs.value
        #     logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
        #     mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
        #     start = 0
        #     end = start + response_length
        #     logprobs, values_pred, mask = (
        #         logprobs[:, start:end],
        #         values_pred[:, start:end],
        #         mask[:, start + 1 : end + 1],
        #     )
        # else:
        start = query_tensors.shape[1] - 1
        tokens, attention_mask, position_ids, outputs, logits, ref_logits, logprobs, ref_logprobs, logratio = [], [], [], [], [], [], [], [], []
        for i in range(num_responses_per_query):
            tokens.append(torch.cat((query_tensors, response_tensors[i]), dim=1))
            attention_mask.append(tokens[i].not_equal(self.tokenizer.pad_token_id).long().to(tokens[i].device))
            position_ids.append(attention_mask[i].long().cumsum(-1) - 1)
            position_ids[i].masked_fill_(attention_mask[i] == 0, 1)
            outputs.append(self.model(tokens[i], attention_mask[i], return_dict=True, position_ids=position_ids[i]))
            logits.append(outputs[i].logits)
            ref_logits.append(self.model.forward_hydra(tokens[i], attention_mask[i], return_dict=True, position_ids=position_ids[i]).logits)

            logprobs.append(logprobs_of_labels(logits[i][:, :-1, :], tokens[i][:, 1:])[:, start:] * attention_mask[i][:, start:-1])
            logprob_len = logprobs[i].shape[1]
            old_logprobs[i] = old_logprobs[i][:, :logprob_len]
            assert logprobs[i].shape == old_logprobs[i].shape
            ref_logprobs.append(logprobs_of_labels(ref_logits[i][:, :-1, :], tokens[i][:, 1:])[:, start:] * attention_mask[i][:, start:-1])
            logratio.append(logprobs[i] - ref_logprobs[i].detach())

        start = query_tensors.shape[1] - 1

        loss, stats = self.p3o_loss(
            logratio=[torch.sum(logratio[i], dim=1) for i in range(num_responses_per_query)],
            rewards=[rewards[i].detach() for i in range(num_responses_per_query)],
            old_logratio=[old_logratios[i].detach() for i in range(num_responses_per_query)],
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            masks=[attention_mask[i][:, start:-1] for i in range(num_responses_per_query)],
        )

        return loss, stats

    def p3o_loss(
        self,
        logratio,
        rewards,
        old_logratio,
        logprobs,
        old_logprobs,
        masks,
    ):
        kl_coef, cliprange_ratio, cliprange, clip_tokenwise, avg_tokenwise, scale_q = (
            self.config.method.kl_coef,
            self.config.method.cliprange_ratio,
            self.config.method.cliprange,
            self.config.method.clip_tokenwise,
            self.config.method.avg_tokenwise,
            self.config.method.scale_q,
        )
        n = masks[0].sum() + masks[1].sum()
        q_diff = (rewards[0] - rewards[1] - kl_coef * (logratio[0] - logratio[1])).detach()
        if scale_q:
            q_diff = q_diff / torch.std(q_diff)
        ratio = torch.exp((logratio[0] - old_logratio[0]) + (logratio[1] - old_logratio[1])).detach()
        cliped_ratio_old = torch.clamp(ratio, 1 / cliprange_ratio, cliprange_ratio)

        if not clip_tokenwise:
            loss1 = -q_diff * cliped_ratio_old.detach() * (logratio[0] - logratio[1])
            loss1_clip = (
                -q_diff
                * cliped_ratio_old.detach()
                * torch.clamp(logratio[0] - logratio[1], old_logratio[0] - old_logratio[1] - cliprange, old_logratio[0] - old_logratio[1] + cliprange)
            )
            if not avg_tokenwise:
                loss = torch.max(loss1, loss1_clip).mean() / 2
            else:
                loss = torch.sum(torch.max(loss1, loss1_clip)) / n
            policy_clipfrac = torch.sum((loss1_clip > loss1).float()) / loss1_clip.shape[0]
        else:
            weights = (-q_diff * cliped_ratio_old.detach()).unsqueeze(-1)
            logprobs[0] = logprobs[0] * masks[0]
            logprobs[1] = logprobs[1] * masks[1]
            old_logprobs[0] = old_logprobs[0] * masks[0]
            old_logprobs[1] = old_logprobs[1] * masks[1]
            loss0 = weights * logprobs[0]
            loss0_cl = weights * torch.clamp(logprobs[0], old_logprobs[0] - cliprange, old_logprobs[0] + cliprange)
            loss1 = -weights * logprobs[1]
            loss1_cl = -weights * torch.clamp(logprobs[1], old_logprobs[1] - cliprange, old_logprobs[1] + cliprange)
            loss = (torch.sum(torch.max(loss0, loss0_cl)) + torch.sum(torch.max(loss1, loss1_cl))) / n

            # log quantity of interest
            policy_clipfrac = (torch.sum((loss0_cl > loss0).float() * masks[0]) + torch.sum((loss1_cl > loss1).float() * masks[1])) / n

        logratio_chosen_mean = torch.mean(logratio[0] * (rewards[0] > rewards[1]).float() + logratio[1] * (rewards[1] > rewards[0]).float()).item()
        logratio_lose_mean = torch.mean(logratio[0] * (rewards[0] < rewards[1]).float() + logratio[1] * (rewards[1] < rewards[0]).float()).item()

        return loss, {
            "loss": loss.item(),
            "reward_diff_mean": torch.mean(rewards[0] - rewards[1]).item(),
            "reward_diff_abs": torch.mean(torch.abs(rewards[0] - rewards[1])).item(),
            "reward_diff_std": torch.std(rewards[0] - rewards[1]).item(),
            "logratio_chosen_mean": logratio_chosen_mean,
            "logratio_lose_mean": logratio_lose_mean,
            "logratio_gap_mean": logratio_chosen_mean - logratio_lose_mean,
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "ratio_min": ratio.min().item(),
            "policy_clipfrac": policy_clipfrac.item(),
            "q_diff_abs_mean": torch.mean(torch.abs(q_diff)).item(),
            "q_diff_mean": torch.mean(q_diff).item(),
            "q_diff_std": torch.std(q_diff).item(),
        }

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        pass

    def create_train_dataloader(self):
        return self.store.create_loader(self.config.train.batch_size, shuffle=True)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.eval_batch_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)

        self.make_experience(self.config.method.num_rollouts)

        self.train_dataloader = self.create_train_dataloader()

        self.n_inner_epochs = self.config.method.p3o_epochs
        self.total_steps = self.config.train.epochs * self.n_inner_epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        p3o_rl_elements = []
        accumulated_stats = []

        while len(p3o_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            # batch = {}
            # for k, v in batch1.items():
            #     batch[k] = torch.vstack([batch1[k], batch1[k]])  # double the prompts, as we want to generate 2 responses per prompt
            prompt_tensors = torch.vstack([batch.input_ids, batch.input_ids])
            attention_mask = torch.vstack([batch.attention_mask, batch.attention_mask])
            samples = self.generate(prompt_tensors, attention_mask)
            stats["time/rollout_generate"] = time() - rollout_generate_time

            # prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(samples, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False)
            padded_prompts = self.accelerator.pad_across_processes(prompt_tensors, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False)
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            # metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=False
                )

                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample
                # NOTE: all_scores[0][i] is the reward due to token (action) i in prompt + response (b/c of how kl is computed)
                all_scores = self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    tokenizer=self.tokenizer,
                    # **metadata,
                )
                # print("373:", all_scores)
                all_scores = [torch.tensor(score, dtype=torch.float, device=device).view(-1) for score in all_scores]
                # Pad 0 reward on the ends
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                # print("377:", all_scores)
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)

                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
                # print("383:", all_scores)
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            scores_mask = scores != -np.inf

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(dim=1).std()
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    # values = outputs.value
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                with torch.no_grad():
                    logits, *_ = self.model(all_tokens, attention_mask=attention_mask, position_ids=position_ids)
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                        ref_logits = ref_logits.to(device)

            if self.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.sum() / max(attention_mask[:, :-1].sum(), 1)
            mean_kl = kl.sum(1).mean()

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            # values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the end of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            logratios = [log_ratio[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start:] for ix in range(n_samples)]

            rollout_count = 0

            for sample_idx in range(n_samples // 2):
                scalar_rewards = torch.tensor([scores[sample_idx][0].cpu(), scores[sample_idx + n_samples // 2][0].cpu()])
                response_tensor = torch.stack([sample_outputs[sample_idx], sample_outputs[sample_idx + n_samples // 2]])
                logratio_sum = torch.tensor([logratios[sample_idx].sum(), logratios[sample_idx + n_samples // 2].sum()])
                logprobs_stack = torch.stack([all_logprobs[sample_idx], all_logprobs[sample_idx + n_samples // 2]])
                assert torch.equal(prompt_tensors[sample_idx], prompt_tensors[sample_idx + n_samples // 2])
                p3o_rl_elements.append(
                    P3ORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=response_tensor,
                        logratios=logratio_sum,
                        logprobs=logprobs_stack,
                        scalar_rewards=scalar_rewards,
                    )
                )
                assert response_tensor.shape[0] == 2

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(p3o_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.config.method.kl_coef
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(p3o_rl_elements)

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """
        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")

        self.accelerator.wait_for_everyone()

        # Save only the base model, so that is could be loaded directly
        # with Hugging Face's `from_pretrained` method
        state_dict = self.accelerator.get_state_dict(self.model.base_model)

        self.accelerator.unwrap_model(self.model).save_pretrained(
            directory,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)
