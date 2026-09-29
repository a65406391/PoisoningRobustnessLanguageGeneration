import os
import time

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from experiments.save_utils import write_results_to_file

from .alignment_certifier import AlignmentCertifier
from .inference import aggregate_robustness_radii_to_dict
from .solver import certify_batch_targeted_attacks


class LanguageGenerationValidityCertifier(AlignmentCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict | None = None):
        super().__init__(hyperparams, device, save_kwargs)

    def multi_sample_robustness_row(
        self,
        ks_poison: list[int],
        q: int,
        preference_test_set: Dataset = None,
        avoid_sentence: str = None,
        batch_size_gen: int = 64,
        batch_size_attack: int = 50,
    ) -> np.ndarray:
        test_set = self.test_set if preference_test_set is None else preference_test_set
        prompts = test_set.get_as_column("prompt")
        avoid_sentences = [avoid_sentence for _ in range(len(prompts))] if avoid_sentence is not None else test_set.get_as_column("rejected")
        tokenized_avoid_sentences = [self.tokenizer.encode(avoid_sentence, add_special_tokens=False) for avoid_sentence in avoid_sentences]

        all_tokenized_responses = []
        gen_start_time = time.perf_counter()
        for partition_idx in range(self.num_partitions):
            logger.info(f"Generating responses for partition {partition_idx+1}/{self.num_partitions} and model {self.llm_type}")
            _, tokenized_responses, _ = self.generate_single_response_iterative(partition_idx, prompts, avoid_sentences, q, batch_size=batch_size_gen)
            all_tokenized_responses.append(tokenized_responses)
        gen_elapsed = time.perf_counter() - gen_start_time
        logger.info(f"[TIMING] tokens*prompts generation: {gen_elapsed:.2f}s ({self.num_partitions} partitions x {len(prompts)} prompts)")

        num_test_samples = len(prompts)
        agg_margin, reduce_classes = self._get_set_rob_radius_against_targeted_attack(
            tokenized_avoid_sentences, all_tokenized_responses, q, num_test_samples, return_gap=True
        )

        partition_dset_size = len(self.train_set) // self.num_partitions
        num_batches = num_test_samples // batch_size_attack
        k_poison_robustness = [[[] for _ in range(num_batches)] for _ in range(q + 1)]
        agg_thresh = int(batch_size_attack * 0.8)
        # Timing accumulators
        milp_times_per_token = [[] for _ in range(q + 1)]  # per-token MILP solve times
        milp_times_per_batch = [0.0 for _ in range(num_batches)]  # total MILP time per batch
        milp_all_stats = []  # all MILP stats for complexity reporting
        batch_wall_times = [0.0 for _ in range(num_batches)]  # wall clock per batch
        certification_start_time = time.perf_counter()
        for q_idx in range(q + 1):
            batch_predictions_per_partition = torch.zeros(
                (num_batches, batch_size_attack, self.num_partitions), device=self.device, dtype=torch.int32
            )
            batch_agg_margins = torch.zeros((num_batches, batch_size_attack), device=self.device, dtype=torch.int32)
            batch_avoid_tokens = torch.zeros((num_batches, batch_size_attack), device=self.device, dtype=torch.int32)
            # First collect the predictions and aggregation margins for each token position
            sample_idx, cnt_pred = 0, 0
            for batch_idx in range(num_batches):
                batch_start_time = time.perf_counter()
                batch_reduce_classes = []
                cnt_pred = 0
                while cnt_pred < batch_size_attack:
                    if sample_idx >= num_test_samples:
                        logger.info(
                            f"Not enough samples for full batch at batch {batch_idx}, token position {q_idx}, only {cnt_pred} samples collected."
                        )
                        break
                    avoid_tokens = tokenized_avoid_sentences[sample_idx]
                    if len(avoid_tokens) <= q_idx:
                        sample_idx += 1
                        continue

                    # Because it possible that all_tok_resp[partition_idx][q_idx] does not exist
                    def __get_token(p, q, s):
                        try:
                            return all_tokenized_responses[p][q][s][0]
                        except IndexError:
                            return -1

                    batch_predictions_per_partition[batch_idx][cnt_pred] = (
                        torch.tensor([__get_token(partition_idx, q_idx, sample_idx) for partition_idx in range(self.num_partitions)])
                        .to(device=self.device, dtype=torch.int32)
                        .view(self.num_partitions)
                    )
                    batch_agg_margins[batch_idx][cnt_pred] = agg_margin[sample_idx][q_idx]
                    batch_avoid_tokens[batch_idx][cnt_pred] = avoid_tokens[q_idx]
                    batch_reduce_classes.append(reduce_classes[sample_idx][q_idx])
                    sample_idx += 1
                    cnt_pred += 1

                if cnt_pred >= agg_thresh:
                    for kp_idx, k_poison in enumerate(ks_poison):
                        robustness_radius, milp_stats = certify_batch_targeted_attacks(
                            batch_predictions_per_partition[batch_idx][:cnt_pred],
                            batch_agg_margins[batch_idx][:cnt_pred],
                            batch_avoid_tokens[batch_idx][:cnt_pred],
                            partition_dset_size,
                            batch_reduce_classes,
                            k_poison,
                            self.device,
                        )
                        k_poison_robustness[q_idx][batch_idx].append(float(robustness_radius))
                        milp_times_per_token[q_idx].append(milp_stats["solve_time"])
                        milp_times_per_batch[batch_idx] += milp_stats["solve_time"]
                        milp_all_stats.append(milp_stats)
                batch_wall_time = time.perf_counter() - batch_start_time
                batch_wall_times[batch_idx] += batch_wall_time

        certification_elapsed = time.perf_counter() - certification_start_time
        total_milp_time = sum(milp_times_per_batch)
        total_milp_solves = len(milp_all_stats)

        # Log timing summary
        logger.info("=" * 60)
        logger.info("[TIMING SUMMARY] Multi-Sample Robustness Row")
        logger.info(f"  Generation (tokens*prompts): {gen_elapsed:.2f}s")
        logger.info(f"  Certification total wall:    {certification_elapsed:.2f}s")
        logger.info(f"  Total MILP solve time:       {total_milp_time:.2f}s ({total_milp_solves} solves)")
        if total_milp_solves > 0:
            logger.info(f"  Avg MILP solve time:         {total_milp_time / total_milp_solves:.4f}s")
        for q_idx in range(q + 1):
            if milp_times_per_token[q_idx]:
                avg_t = np.mean(milp_times_per_token[q_idx])
                logger.info(f"  Token {q_idx}: avg MILP={avg_t:.4f}s ({len(milp_times_per_token[q_idx])} solves)")
        for batch_idx in range(num_batches):
            logger.info(f"  Batch {batch_idx}: MILP={milp_times_per_batch[batch_idx]:.2f}s, wall={batch_wall_times[batch_idx]:.2f}s")
        if milp_all_stats:
            s0 = milp_all_stats[0]
            logger.info(
                f"  MILP complexity (first solve): vars={s0['num_vars']}, constrs={s0['num_constrs']}, "
                f"int_vars={s0['num_int_vars']}, gen_constrs={s0['num_gen_constrs']}"
            )
        logger.info("=" * 60)

        avg_wc_acc_per_q_and_poison_budget = {}
        for q_idx in range(q):
            avg_wc_acc_q = np.zeros((len(ks_poison),))
            for kp_idx, k_poison in enumerate(ks_poison):
                try:
                    batch_results = [
                        k_poison_robustness[q_idx][b_idx][kp_idx]
                        for b_idx in range(len(k_poison_robustness[q_idx]))
                        if len(k_poison_robustness[q_idx][b_idx]) > 0
                    ]
                    avg_wc_acc_q[kp_idx] = np.mean(batch_results) if len(batch_results) > 0 else 0.0
                except Exception as e:
                    logger.error(f"Error computing average wc acc for q_idx {q_idx}, k_poison {k_poison}: {e}. Setting to 0.0")
                    avg_wc_acc_q[kp_idx] = 0.0
            avg_wc_acc_q_per_poison_budget = {int(k_poison): float(avg_wc_acc_q[k_poison_idx]) for k_poison_idx, k_poison in enumerate(ks_poison)}
            avg_wc_acc_per_q_and_poison_budget[q_idx] = avg_wc_acc_q_per_poison_budget
        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            new_res_file = rf + f"_{self.llm_type}" + ext
            category = self.method_name + f"_valid_generation_row_partitions_{self.num_partitions}_q_{q}_msc_batch_size_{batch_size_attack}"
            timing_info = {
                "generation_time_s": round(gen_elapsed, 2),
                "certification_wall_time_s": round(certification_elapsed, 2),
                "total_milp_solve_time_s": round(total_milp_time, 2),
                "num_milp_solves": total_milp_solves,
                "avg_milp_solve_time_s": round(total_milp_time / max(total_milp_solves, 1), 4),
                "per_batch_wall_time_s": [round(t, 2) for t in batch_wall_times],
                "per_batch_milp_time_s": [round(t, 2) for t in milp_times_per_batch],
                "per_token_avg_milp_time_s": {
                    q_idx: round(float(np.mean(milp_times_per_token[q_idx])), 4) for q_idx in range(q + 1) if milp_times_per_token[q_idx]
                },
            }
            if milp_all_stats:
                timing_info["milp_complexity"] = {
                    "num_vars": milp_all_stats[0]["num_vars"],
                    "num_constrs": milp_all_stats[0]["num_constrs"],
                    "num_int_vars": milp_all_stats[0]["num_int_vars"],
                    "num_gen_constrs": milp_all_stats[0]["num_gen_constrs"],
                }
            write_results_to_file(
                new_res_file,
                {
                    "certified_robustness_per_token_idx_and_k_poison": avg_wc_acc_per_q_and_poison_budget,
                    "timing": timing_info,
                },
                category,
            )

        return k_poison_robustness

    def vote_and_get_robustness_row(
        self, q: int, preference_test_set: Dataset = None, avoid_sentence: str = None, batch_size=64, phrase_len: int = 1
    ) -> np.ndarray:
        test_set = self.test_set if preference_test_set is None else preference_test_set
        prompts = test_set.get_as_column("prompt")
        avoid_sentences = [avoid_sentence for _ in range(len(prompts))] if avoid_sentence is not None else test_set.get_as_column("rejected")
        tokenized_avoid_sentences = [self.tokenizer.encode(avoid_sentence, add_special_tokens=False) for avoid_sentence in avoid_sentences]

        all_tokenized_responses = []
        for partition_idx in range(self.num_partitions):
            # Checkpoint the generated tokens per partition, so a crash does not throw away the previous partitions
            ckpt_file = os.path.join(
                AlignmentCertifier.RESULT_DIR, self.save_load_dir, f"partition_{partition_idx}", f"iterative_responses_q_{q}_phrase_len_{phrase_len}_n_{len(prompts)}.pt"
            )
            if os.path.exists(ckpt_file):
                logger.info(f"Loading generated responses for partition {partition_idx} from {ckpt_file}")
                tokenized_responses = torch.load(ckpt_file)
            else:
                _, tokenized_responses, _ = self.generate_single_response_iterative(
                    partition_idx, prompts, avoid_sentences, q, batch_size=batch_size, phrase_len=phrase_len
                )
                torch.save(tokenized_responses, ckpt_file)
            all_tokenized_responses.append(tokenized_responses)

        num_test_samples = len(prompts)
        rob_radius = self._get_set_rob_radius_against_targeted_attack(
            tokenized_avoid_sentences, all_tokenized_responses, q, num_test_samples, phrase_len=phrase_len
        )

        rob_radius = rob_radius.permute(1, 0)  # [q+1, num_test_samples]
        step_indices = range(0, q + 1, phrase_len)
        robustness_dict_per_token = {i: aggregate_robustness_radii_to_dict(rob_radius[i]) for i, step_idx in enumerate(step_indices)}
        # take the aggregation margin min over dimension 0 (q + 1) to result in a [num_test_samples] tensor
        min_over_q_rob_radius = torch.min(rob_radius, dim=0).values
        assert min_over_q_rob_radius.shape[0] == num_test_samples, "Aggregation margin shape does not match number of test samples."
        rob_certificate_prompt = aggregate_robustness_radii_to_dict(min_over_q_rob_radius)
        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            new_res_file = rf + f"_{self.llm_type}" + ext
            category = self.method_name + f"_valid_generation_row_partitions_{self.num_partitions}_q_{q}_phrase_len_{phrase_len}"
            cert_key = f"certified_robustness_per_{'phrase' if phrase_len > 1 else 'token'}_idx"
            write_results_to_file(
                new_res_file,
                {
                    cert_key: robustness_dict_per_token,
                    "robustness_prompt": rob_certificate_prompt,
                },
                category,
            )

        return rob_radius.cpu().numpy()

    def _get_set_rob_radius_against_targeted_attack(
        self,
        tokenized_avoid_sentences: list[list[int]],
        all_tokenized_responses: list[list[list[int]]],
        q: int,
        num_test_samples: int,
        return_gap: bool = False,
        phrase_len: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, list[int]]:

        step_indices = range(0, q + 1, phrase_len)
        num_steps = len(step_indices)
        generalized_margin_vec = torch.zeros((num_test_samples, num_steps), device=self.device, dtype=torch.int32)
        reduce_classes = [[] for _ in range(num_test_samples)]
        for test_sample_idx in range(num_test_samples):
            curr_avoid_tokens = tokenized_avoid_sentences[test_sample_idx]
            effective_q = min(q + 1, len(curr_avoid_tokens))
            for i, q_idx in enumerate(step_indices):
                if q_idx >= effective_q:
                    last_val = []
                    # Last Observation Carried Forward - use the last available robustness radius
                    if i > 0:
                        generalized_margin_vec[test_sample_idx, i] = generalized_margin_vec[test_sample_idx, i - 1]
                    if len(reduce_classes[test_sample_idx]) > 0:
                        last_val = reduce_classes[test_sample_idx][-1]
                    reduce_classes[test_sample_idx].append(last_val)
                    continue

                curr_phrase_len = min(phrase_len, effective_q - q_idx)
                # We only care about the phrase_len tokens at each reprompting because that is the one being voted on
                # curr_phrases = torch.tensor(
                #     [all_tokenized_responses[partition_idx][i][test_sample_idx][:curr_phrase_len] for partition_idx in range(self.num_partitions)]
                # ).to(device=self.device, dtype=torch.int32)
                curr_step_responses = []
                for partition_idx in range(self.num_partitions):
                    partition_responses = all_tokenized_responses[partition_idx]
                    if i < len(partition_responses):
                        curr_step_responses.append(partition_responses[i][test_sample_idx][:curr_phrase_len])
                    else:
                        curr_step_responses.append([])
                curr_phrases = [tuple(p) for p in curr_step_responses]
                phrase_to_id, vote_ids = {}, []
                for p in curr_phrases:
                    if p not in phrase_to_id:
                        phrase_to_id[p] = len(phrase_to_id)
                    vote_ids.append(phrase_to_id[p])
                word_token_votes = torch.bincount(torch.tensor(vote_ids, device=self.device), minlength=len(phrase_to_id))

                num_unique_tokens = torch.sum(word_token_votes > 0).item()
                topk = torch.topk(word_token_votes, k=num_unique_tokens)
                topk_classes, topk_votes = topk.indices, topk.values

                avoid_phrase = tuple(tokenized_avoid_sentences[test_sample_idx][q_idx : q_idx + curr_phrase_len])
                avoid_id = phrase_to_id.get(avoid_phrase, -1)
                # avoid_token_match = topk_classes == tokenized_avoid_sentences[test_sample_idx][q_idx]
                avoid_token_match = topk_classes == avoid_id
                num_matches = torch.sum(avoid_token_match).item()
                start_votes_sensitive_tok = 0
                if num_matches > 0:
                    # Find first match, skip if avoid token is the predicted token
                    match_idx = torch.nonzero(avoid_token_match, as_tuple=False)[0].item()
                    if match_idx == 0:
                        generalized_margin_vec[test_sample_idx, i] = 0
                        reduce_classes[test_sample_idx].append([])
                        continue
                    start_votes_sensitive_tok = topk_votes[match_idx].item()
                    # Remove the matching token from topk_votes and topk_classes
                    topk_votes = torch.cat((topk_votes[:match_idx], topk_votes[match_idx + 1 :]))
                    topk_classes = torch.cat((topk_classes[:match_idx], topk_classes[match_idx + 1 :]))
                try:
                    # Be conservative for now for phrase-level (we can do a lexicographic comparison later, if needed)
                    target_class = int(tokenized_avoid_sentences[test_sample_idx][q_idx]) if phrase_len == 1 else -1
                    if return_gap:
                        generalized_margin_vec[test_sample_idx, i], rc = self._gap_avoid_token(
                            topk_votes, topk_classes, start_votes_sensitive_tok, target_class
                        )
                        reduce_classes[test_sample_idx].append(rc)
                    else:
                        generalized_margin_vec[test_sample_idx, i] = self._rob_radius_avoid_token(
                            topk_votes, topk_classes, start_votes_sensitive_tok, target_class
                        )
                except Exception as e:
                    logger.error(f"Error computing agg margin for test sample {test_sample_idx}, q_idx {q_idx}: {e}")
                    generalized_margin_vec[test_sample_idx, i] = 0
                    reduce_classes[test_sample_idx].append([])
        if return_gap:
            return generalized_margin_vec, reduce_classes
        return generalized_margin_vec

    def _rob_radius_avoid_token(self, topk_votes: torch.Tensor, topk_classes: torch.Tensor, target_initial_votes: int, target_class: int) -> int:
        topk_votes= torch.cat([topk_votes, topk_votes.new_zeros(1)])  # Append a zero to handle the last case

        delta_ii = (topk_votes[:-1] - topk_votes[1:]) * torch.arange(1, len(topk_votes)).to(device=self.device)
        ct_prime = torch.cumsum(delta_ii, 0) + target_initial_votes
        mask = ct_prime > topk_votes[1:]

        if not mask.any():
            s = len(ct_prime) - 1
        else:
            s = mask.to(dtype=torch.int32).argmax().item()

        prev = target_initial_votes if s == 0 else ct_prime[s - 1]

        tie = int(topk_classes[0] > target_class)
        rob_radius = (prev - target_initial_votes) + torch.floor((topk_votes[s] - prev + 1 + tie) * (s + 1) / (s + 2)) - 1
        return int(rob_radius)

    def _gap_avoid_token(
        self, topk_votes: torch.Tensor, topk_classes: torch.Tensor, target_initial_votes: int, target_class
    ) -> tuple[int, list[int]]:
        if len(topk_votes) < 2:
            # This is equivalent to DPA
            return int(topk_votes[0]) - target_initial_votes + (int(topk_classes[0]) < target_class), [int(topk_classes[0])]

        delta_ii = (topk_votes[:-1] - topk_votes[1:]) * torch.arange(1, len(topk_votes)).to(device=self.device)
        ct_prime = torch.cumsum(delta_ii, 0) + target_initial_votes
        mask = ct_prime > topk_votes[1:]

        if not mask.any():
            s = len(ct_prime) - 1
        else:
            s = mask.to(dtype=torch.int32).argmax().item()

        if s == 0:
            # This is equivalent to DPA
            return int(topk_votes[0]) - target_initial_votes + (int(topk_classes[0]) < target_class), [int(topk_classes[0])]

        exact_meet = float(ct_prime[s - 1]) + (float(topk_votes[s]) - float(ct_prime[s - 1])) * (s + 1) / (s + 2)
        gap = [float(tvi) - exact_meet for tvi in topk_votes[: s + 1]] + [exact_meet - target_initial_votes]

        return round(sum(gap)), [int(tvi) for tvi in topk_classes[: s + 1]]
