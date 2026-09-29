import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import trange

from experiments.save_utils import write_results_to_file

from .alignment_certifier import AlignmentCertifier
from .certification_methods import AggregationType
from .inference import aggregate_robustness_radii_to_dict
from .solver import certify_batch_dpa, numpy_certify_batch_roe


class LanguageGenerationStabilityCertifier(AlignmentCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict | None = None):
        super().__init__(hyperparams, device, save_kwargs)

    def multi_sample_robustness_column(
        self, ks_poison: list[int], q: int, agg_type: AggregationType, preference_test_set: Dataset = None, batch_size_gen=64, batch_size_attack=64
    ) -> np.ndarray:
        test_set = self.test_set if preference_test_set is None else preference_test_set
        all_tokenized_responses, all_logits = [], []
        gen_start_time = time.perf_counter()
        for partition_idx in range(self.num_partitions):
            logger.info(f"Generating responses for partition {partition_idx+1}/{self.num_partitions}")
            decoded_responses, tokenized_responses, logits_responses = self.generate_responses(partition_idx, test_set, batch_size=batch_size_gen)
            all_tokenized_responses.append(tokenized_responses)
            logits_responses = np.stack(logits_responses)
            all_logits.append(logits_responses)

            # Save the generated responses to a CSV file
            df = pd.DataFrame({"response": decoded_responses, "tokenized_response": tokenized_responses})
            response_save_path = os.path.join(
                AlignmentCertifier.RESULT_DIR, self.save_load_dir, f"responses_partition_{partition_idx}_agg_{agg_type}_msc.csv"
            )
            df.to_csv(response_save_path, index=False)
            logger.info(f"Saved generated responses to {response_save_path}")
            gc.collect()
            torch.cuda.empty_cache()
        gen_elapsed = time.perf_counter() - gen_start_time
        logger.info(f"[TIMING] tokens*prompts generation: {gen_elapsed:.2f}s ({self.num_partitions} partitions x {len(test_set)} prompts)")
        all_logits = np.array(all_logits)
        print(f"all_logits shape: {all_logits.shape}")

        num_generated = (len(test_set) // batch_size_gen) * batch_size_gen
        num_test_samples = (min(num_generated, len(test_set)) // batch_size_attack) * batch_size_attack  # Ignore remainder for simplicity
        partition_dset_size = len(self.train_set) // self.num_partitions
        num_batches = num_test_samples // batch_size_attack
        if num_batches == 0:
            raise ValueError(
                f"No batches to certify: num_test_samples={num_test_samples}, batch_size_attack={batch_size_attack}. "
                "Try increasing num_test_points or decreasing batch_size_attack/batch_size_gen."
            )
        cs_pred, cs_sec, cs_third, votes_pred, votes_sec, votes_third = tuple(
            [torch.zeros((q, num_test_samples), device=self.device, dtype=torch.int32) for _ in range(6)]
        )
        predicted_tokens_per_partition = torch.zeros((num_test_samples, self.num_partitions, q), device=self.device, dtype=torch.int32)
        for test_sample_idx in range(num_test_samples):
            predicted_tokens_cut_to_q = [all_tokenized_responses[partition_idx][test_sample_idx][:q] for partition_idx in range(self.num_partitions)]
            for partition_idx in range(self.num_partitions):
                predicted_tokens_per_partition[test_sample_idx] = torch.tensor(predicted_tokens_cut_to_q, device=self.device, dtype=torch.int32)

            predicted_tokens_cut_to_q = torch.tensor(predicted_tokens_cut_to_q, device=self.device, dtype=torch.int32).permute(1, 0)
            for q_idx in range(q):
                word_tokens = predicted_tokens_cut_to_q[q_idx]
                word_token_votes = torch.bincount(word_tokens, minlength=self.tokenizer.vocab_size)

                top3 = torch.topk(word_token_votes, k=3)
                top3_classes, top3_votes = top3.indices, top3.values
                cs_pred[q_idx, test_sample_idx], votes_pred[q_idx, test_sample_idx] = top3_classes[0], top3_votes[0]
                if len(top3_classes) > 1:
                    cs_sec[q_idx, test_sample_idx], votes_sec[q_idx, test_sample_idx] = top3_classes[1], top3_votes[1]
                if len(top3_classes) > 2:
                    cs_third[q_idx, test_sample_idx], votes_third[q_idx, test_sample_idx] = top3_classes[2], top3_votes[2]

        k_poison_robustness = np.zeros((q, num_batches, len(ks_poison)), dtype=float)
        # Timing accumulators: milp_times[q_idx][batch_idx] = list of solve times for each k_poison
        milp_times_per_token = [[] for _ in range(q)]  # per-token MILP solve times
        milp_times_per_batch = [0.0 for _ in range(num_batches)]  # total MILP time per batch
        milp_all_stats = []  # all MILP stats for complexity reporting
        batch_wall_times = []  # wall clock per batch
        intrinsic_robustness = np.zeros((self.num_partitions, batch_size_attack))
        predicted_tokens_per_partition = predicted_tokens_per_partition.permute(2, 0, 1)  # [q, num_samples, num_partitions]
        num_classes = self.tokenizer.vocab_size
        roe_logits, roe_batch_labels = None, None
        if agg_type == AggregationType.ROE:
            # Compute this only once
            roe_logits = self._extract_roe_logits(all_logits, cs_pred, cs_sec, q, num_test_samples)
            # roe_logits shape is [num_partitions, q, num_test_samples, num_classes]
            _, roe_batch_labels = self._aggregate_roe(cs_pred, cs_sec, votes_pred, votes_sec, votes_third, roe_logits, return_prediction=True)

        certification_start_time = time.perf_counter()
        for batch_idx in trange(num_batches, desc="Multi Sample Certification"):
            batch_start_time = time.perf_counter()
            start, end = batch_idx * batch_size_attack, (batch_idx + 1) * batch_size_attack
            for q_idx in range(q):
                #! We assume, for now, that the correct predictions is the most voted token (using DPA)
                batch_preds = predicted_tokens_per_partition[q_idx, start:end, :]
                match agg_type:
                    case AggregationType.DPA:
                        batch_labels = cs_pred[q_idx, start:end]
                        for kp_idx, k_poison in enumerate(ks_poison):
                            robustness_radii, milp_stats = certify_batch_dpa(
                                intrinsic_robustness, batch_preds, batch_labels, num_classes, partition_dset_size, k_poison, self.device
                            )
                            k_poison_robustness[q_idx, batch_idx, kp_idx] = robustness_radii
                            milp_times_per_token[q_idx].append(milp_stats["solve_time"])
                            milp_times_per_batch[batch_idx] += milp_stats["solve_time"]
                            milp_all_stats.append(milp_stats)
                    case AggregationType.ROE:
                        batch_actual_size = end - start
                        batch_preds_np = batch_preds.cpu().numpy()
                        # Fill top-k logits and compute ROE predictions
                        batch_logits_per_partition = np.full((batch_actual_size, self.num_partitions, num_classes), -30000.0, dtype=np.float32)
                        for b_idx in range(batch_actual_size):
                            for p_idx in range(self.num_partitions):
                                logits_vals, logits_idxs = all_logits[p_idx][start + b_idx][q_idx]
                                batch_logits_per_partition[b_idx, p_idx, logits_idxs.astype(int)] = logits_vals
                        # Get the "true" labels: we assume, for now, that the correct predictions is the most voted token (using ROE)
                        batch_labels_np = roe_batch_labels[q_idx, start:end].cpu().numpy()
                        self.log_roe_msc(
                            cs_pred,
                            cs_sec,
                            cs_third,
                            votes_pred,
                            votes_sec,
                            votes_third,
                            batch_logits_per_partition,
                            q_idx,
                            start,
                            end,
                            batch_actual_size,
                            batch_idx,
                            batch_labels_np,
                        )
                        for kp_idx, k_poison in enumerate(ks_poison):
                            robustness_radii, milp_stats = numpy_certify_batch_roe(
                                intrinsic_robustness,
                                batch_preds_np,
                                batch_logits_per_partition,
                                batch_labels_np,
                                num_classes,
                                partition_dset_size,
                                k_poison,
                            )
                            k_poison_robustness[q_idx, batch_idx, kp_idx] = robustness_radii
                            milp_times_per_token[q_idx].append(milp_stats["solve_time"])
                            milp_times_per_batch[batch_idx] += milp_stats["solve_time"]
                            milp_all_stats.append(milp_stats)
                        del batch_logits_per_partition
                    case _:
                        raise ValueError(f"Aggregation type {agg_type} not implemented for alignment tasks.")
            batch_wall_time = time.perf_counter() - batch_start_time
            batch_wall_times.append(batch_wall_time)
            logger.info(f"[TIMING] Batch {batch_idx}: wall={batch_wall_time:.2f}s, MILP_solve={milp_times_per_batch[batch_idx]:.2f}s")

        certification_elapsed = time.perf_counter() - certification_start_time
        total_milp_time = sum(milp_times_per_batch)
        total_milp_solves = len(milp_all_stats)

        # Log per-token average MILP time
        logger.info("=" * 60)
        logger.info("[TIMING SUMMARY] Multi-Sample Robustness Column")
        logger.info(f"  Generation (tokens*prompts): {gen_elapsed:.2f}s")
        logger.info(f"  Certification total wall:    {certification_elapsed:.2f}s")
        logger.info(f"  Total MILP solve time:       {total_milp_time:.2f}s ({total_milp_solves} solves)")
        if total_milp_solves > 0:
            logger.info(f"  Avg MILP solve time:         {total_milp_time / total_milp_solves:.4f}s")
        for q_idx in range(q):
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
            avg_wc_acc_q = k_poison_robustness[q_idx].mean(axis=0)
            avg_wc_acc_q_per_poison_budget = {int(k_poison): float(avg_wc_acc_q[k_poison_idx]) for k_poison_idx, k_poison in enumerate(ks_poison)}
            avg_wc_acc_per_q_and_poison_budget[q_idx] = avg_wc_acc_q_per_poison_budget
        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            new_res_file = rf + f"_{self.llm_type}" + ext
            category = (
                self.method_name
                + "_agg_type_"
                + str(agg_type).lower()
                + f"_partitions_{self.num_partitions}"
                + f"_msc_batch_size_{batch_size_attack}"
            )
            timing_info = {
                "generation_time_s": round(gen_elapsed, 2),
                "certification_wall_time_s": round(certification_elapsed, 2),
                "total_milp_solve_time_s": round(total_milp_time, 2),
                "num_milp_solves": total_milp_solves,
                "avg_milp_solve_time_s": round(total_milp_time / max(total_milp_solves, 1), 4),
                "per_batch_wall_time_s": [round(t, 2) for t in batch_wall_times],
                "per_batch_milp_time_s": [round(t, 2) for t in milp_times_per_batch],
                "per_token_avg_milp_time_s": {
                    q_idx: round(float(np.mean(milp_times_per_token[q_idx])), 4) for q_idx in range(q) if milp_times_per_token[q_idx]
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

    def log_roe_msc(
        self,
        cs_pred,
        cs_sec,
        cs_third,
        votes_pred,
        votes_sec,
        votes_third,
        batch_logits_per_partition,
        q_idx,
        start,
        end,
        batch_actual_size,
        batch_idx,
        batch_labels_np,
    ):
        dpa_winners = cs_pred[q_idx, start:end].cpu().numpy()
        roe_winners = batch_labels_np

        mismatches = np.where(dpa_winners != roe_winners)[0]
        print(f"ROE vs DPA Mismatches in this batch: {len(mismatches)}/{batch_actual_size}")
        for b_idx in range(batch_actual_size):
            global_idx = start + b_idx
            print(f"Sample {b_idx}:")
            c1 = cs_pred[q_idx, global_idx]
            c2 = cs_sec[q_idx, global_idx]
            c3 = cs_third[q_idx, global_idx]
            print(f"\nSample {b_idx} (Global {global_idx}):")
            print(f"  DPA Winner: {c1} | ROE Winner: {roe_winners[b_idx]}")
            print(
                f"  Total Votes -> C1 = {c1}: {votes_pred[q_idx, global_idx]}, C2 = {c2}: {votes_sec[q_idx, global_idx]}, C3={c3}: {votes_third[q_idx, global_idx]}"
            )
            for p_idx in range(self.num_partitions):
                l1 = batch_logits_per_partition[b_idx, p_idx, int(c1)]
                l2 = batch_logits_per_partition[b_idx, p_idx, int(c2)]
                l3 = batch_logits_per_partition[b_idx, p_idx, int(c3)]
                print(f"    Part {p_idx} Logits -> {c1}: {l1:.2f}, {c2}: {l2:.2f}, {c3}: {l3:.2f}")

    def vote_and_get_robustness_column(self, q: int, agg_type: AggregationType, preference_test_set: Dataset = None, batch_size=64) -> np.ndarray:
        test_set = self.test_set if preference_test_set is None else preference_test_set
        all_tokenized_responses, all_logits = [], []
        for partition_idx in range(self.num_partitions):
            logger.info(f"Generating responses for partition {partition_idx+1}/{self.num_partitions}")
            decoded_responses, tokenized_responses, logits_responses = self.generate_responses(partition_idx, test_set, batch_size=batch_size)
            all_tokenized_responses.append(tokenized_responses)
            all_logits.append(logits_responses)

            # Save the generated responses to a CSV file
            df = pd.DataFrame({"response": decoded_responses, "tokenized_response": tokenized_responses})
            response_save_path = os.path.join(
                AlignmentCertifier.RESULT_DIR, self.save_load_dir, f"responses_partition_{partition_idx}_agg_{agg_type}.csv"
            )
            df.to_csv(response_save_path, index=False)
            logger.info(f"Saved generated responses to {response_save_path}")

        num_test_samples = (len(test_set) // batch_size) * batch_size  # Ignore remainder for simplicity
        cs_pred, cs_sec, cs_third, votes_pred, votes_sec, votes_third = tuple(
            [torch.zeros((q, num_test_samples), device=self.device, dtype=torch.int32) for _ in range(6)]
        )
        predicted_tokens_per_partition = torch.zeros((num_test_samples, self.num_partitions, q), device=self.device, dtype=torch.int32)
        for test_sample_idx in range(num_test_samples):
            predicted_tokens_cut_to_q = [all_tokenized_responses[partition_idx][test_sample_idx][:q] for partition_idx in range(self.num_partitions)]
            for partition_idx in range(self.num_partitions):
                predicted_tokens_per_partition[test_sample_idx] = torch.tensor(predicted_tokens_cut_to_q, device=self.device, dtype=torch.int32)

            predicted_tokens_cut_to_q = torch.tensor(predicted_tokens_cut_to_q, device=self.device, dtype=torch.int32).permute(1, 0)
            for q_idx in range(q):
                word_tokens = predicted_tokens_cut_to_q[q_idx]
                print(f"Word tokens at test sample {test_sample_idx}, q {q_idx}: {word_tokens}")
                word_token_votes = torch.bincount(word_tokens, minlength=self.tokenizer.vocab_size)

                top3 = torch.topk(word_token_votes, k=3)
                top3_classes, top3_votes = top3.indices, top3.values
                cs_pred[q_idx, test_sample_idx], votes_pred[q_idx, test_sample_idx] = top3_classes[0], top3_votes[0]
                if len(top3_classes) > 1:
                    cs_sec[q_idx, test_sample_idx], votes_sec[q_idx, test_sample_idx] = top3_classes[1], top3_votes[1]
                if len(top3_classes) > 2:
                    cs_third[q_idx, test_sample_idx], votes_third[q_idx, test_sample_idx] = top3_classes[2], top3_votes[2]

        aggregation_margin = None
        match agg_type:
            case AggregationType.DPA:
                aggregation_margin = self._aggregate_dpa(cs_pred, cs_sec, votes_pred, votes_sec)
            case AggregationType.ROE:
                roe_logits = self._extract_roe_logits(all_logits, cs_pred, cs_sec, q, num_test_samples)
                aggregation_margin = self._aggregate_roe(cs_pred, cs_sec, votes_pred, votes_sec, votes_third, roe_logits)
            case _:
                raise ValueError(f"Aggregation type {agg_type} not implemented for alignment tasks.")

        robustness_dict_per_token = {q_idx: aggregate_robustness_radii_to_dict(aggregation_margin[q_idx]) for q_idx in range(q)}
        # take the aggregation margin min over dimension 0 (q) to result in a [num_test_samples] tensor
        min_over_q_agg_margin = torch.min(aggregation_margin, dim=0).values
        assert min_over_q_agg_margin.shape[0] == num_test_samples, "Aggregation margin shape does not match number of test samples."
        rob_certificate_prompt = aggregate_robustness_radii_to_dict(min_over_q_agg_margin)
        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            new_res_file = rf + f"_{self.llm_type}" + ext
            category = self.method_name + "_agg_type_" + str(agg_type).lower() + f"_partitions_{self.num_partitions}"
            write_results_to_file(
                new_res_file,
                {
                    "certified_robustness_per_token_idx": robustness_dict_per_token,
                    "robustness_prompt": rob_certificate_prompt,
                },
                category,
            )

        return aggregation_margin.cpu().numpy()

    def phrase_level_stability(self, num_phrases: int, tokens_per_phrase: int = 3, preference_test_set: Dataset = None, batch_size=64) -> np.ndarray:
        test_set = self.test_set if preference_test_set is None else preference_test_set
        q = num_phrases * tokens_per_phrase
        all_tokenized_responses, all_logits = [], []
        for partition_idx in range(self.num_partitions):
            logger.info(f"Generating responses for partition {partition_idx+1}/{self.num_partitions}")
            decoded_responses, tokenized_responses, logits_responses = self.generate_responses(
                partition_idx, test_set, batch_size=batch_size, max_new_tokens=q * 1.5  # paddina for safety
            )
            all_tokenized_responses.append(tokenized_responses)
            all_logits.append(logits_responses)

            # Save the generated responses to a CSV file
            df = pd.DataFrame({"response": decoded_responses, "tokenized_response": tokenized_responses})
            response_save_path = os.path.join(
                AlignmentCertifier.RESULT_DIR,
                self.save_load_dir,
                f"responses_partition_{partition_idx}_phrase_stability_t_{tokens_per_phrase}_p_{num_phrases}.csv",
            )
            df.to_csv(response_save_path, index=False)
            logger.info(f"Saved generated responses to {response_save_path}")

        num_test_samples = (len(test_set) // batch_size) * batch_size  # Ignore remainder for simplicity
        aggregation_margin = torch.zeros((num_phrases, num_test_samples), device=self.device, dtype=torch.int32)
        for test_sample_idx in range(num_test_samples):
            predicted_tokens_cut_to_q = [all_tokenized_responses[partition_idx][test_sample_idx][:q] for partition_idx in range(self.num_partitions)]

            for phrase_idx in range(num_phrases):
                phrase_to_class_tokens, max_class, class_counts = {}, -1, [0 for _ in range(self.num_partitions)]
                start_idx = phrase_idx * tokens_per_phrase
                end_idx = start_idx + tokens_per_phrase
                for partition_idx in range(self.num_partitions):
                    phrase = tuple(predicted_tokens_cut_to_q[partition_idx][start_idx:end_idx])
                    if phrase not in phrase_to_class_tokens:
                        max_class += 1
                        phrase_to_class_tokens[phrase] = max_class
                    class_counts[phrase_to_class_tokens[phrase]] += 1
                top2 = torch.topk(torch.tensor(class_counts), k=2)
                top2_classes, top2_votes = top2.indices, top2.values
                tie_break = top2_classes[0] > top2_classes[1]
                aggregation_margin[phrase_idx, test_sample_idx] = torch.floor((top2_votes[0] - (top2_votes[1] + tie_break)) / 2)

        robustness_dict_per_token = {ph_idx: aggregate_robustness_radii_to_dict(aggregation_margin[ph_idx]) for ph_idx in range(num_phrases)}
        # take the aggregation margin min over dimension 0 (num_phrases) to result in a [num_test_samples] tensor
        min_over_q_agg_margin = torch.min(aggregation_margin, dim=0).values
        assert (
            min_over_q_agg_margin.shape[0] == num_test_samples
        ), f"Aggregation margin shape does not match number of test samples, got {min_over_q_agg_margin.shape[0]} expected {num_test_samples}."
        rob_certificate_prompt = aggregate_robustness_radii_to_dict(min_over_q_agg_margin)
        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            new_res_file = rf + f"_{self.llm_type}" + ext
            category = self.method_name + f"_phrase_stability_t_{tokens_per_phrase}_p_{num_phrases}" + f"_partitions_{self.num_partitions}"
            write_results_to_file(
                new_res_file,
                {
                    "certified_robustness_per_phrase_idx": robustness_dict_per_token,
                    "robustness_prompt_phrase_level": rob_certificate_prompt,
                },
                category,
            )

        return aggregation_margin.cpu().numpy()

    def _aggregate_dpa(self, c_pred: torch.Tensor, c_sec: torch.Tensor, votes_pred: torch.Tensor, votes_sec: torch.Tensor) -> torch.Tensor:
        q = int(c_pred.shape[0])
        agg_margins = torch.zeros_like(votes_pred)
        for q_idx in range(q):
            tie = (c_sec[q_idx] < c_pred[q_idx]).to(dtype=torch.int32)
            agg_margins[q_idx] = torch.floor((votes_pred[q_idx] - (votes_sec[q_idx] + tie)) / 2).int()

        return agg_margins

    def _extract_roe_logits(self, all_logits: list, cs_pred: torch.Tensor, cs_sec: torch.Tensor, q: int, num_test_samples: int) -> torch.Tensor:
        """Extract logits for top-2 classes from stored top-k logits"""
        roe_logits = torch.zeros((self.num_partitions, q, num_test_samples, 2), device=self.device, dtype=torch.float32)  # [pred_logit, sec_logit]

        for test_sample_idx in range(num_test_samples):
            for q_idx in range(q):
                pred_class = cs_pred[q_idx, test_sample_idx].item()
                sec_class = cs_sec[q_idx, test_sample_idx].item()
                for partition_idx in range(self.num_partitions):
                    vals, idxs = all_logits[partition_idx][test_sample_idx][q_idx]

                    pred_mask = idxs == pred_class
                    roe_logits[partition_idx, q_idx, test_sample_idx, 0] = (
                        vals[pred_mask][0].item() if pred_mask.any() else AlignmentCertifier.NOT_IN_TOPK_THRESHOLD
                    )
                    sec_mask = idxs == sec_class
                    roe_logits[partition_idx, q_idx, test_sample_idx, 1] = (
                        vals[sec_mask][0].item() if sec_mask.any() else AlignmentCertifier.NOT_IN_TOPK_THRESHOLD
                    )

        return roe_logits

    def _aggregate_roe(
        self,
        c_pred: torch.Tensor,
        c_sec: torch.Tensor,
        votes_pred: torch.Tensor,
        votes_sec: torch.Tensor,
        votes_third: torch.Tensor,
        roe_logits: torch.Tensor,
        return_prediction: bool = False,
    ) -> torch.Tensor:
        # 1. Round 1 Certificate (c_pred avoids dropping to 3rd)
        gap_sec = (votes_pred - votes_sec + 2) // 2
        gap_third = (votes_pred - votes_third + 2) // 2
        cert_r1 = torch.clamp(gap_sec + gap_third, min=0)

        # 2.1. Round 2 Elections (Strict Head-to-head)
        has_data_c1 = roe_logits[..., 0] > AlignmentCertifier.NOT_IN_TOPK_THRESHOLD
        has_data_c2 = roe_logits[..., 1] > AlignmentCertifier.NOT_IN_TOPK_THRESHOLD

        # Tally C1 Votes
        c1_better = (roe_logits[..., 0] > roe_logits[..., 1]) & has_data_c1
        c1_tie_break = has_data_c1 & has_data_c2 & (roe_logits[..., 0] == roe_logits[..., 1]) & (c_pred < c_sec)
        vote_c1 = c1_better | c1_tie_break
        binary_votes_c1 = vote_c1.sum(dim=0)

        # Tally C2 Votes
        c2_better = (roe_logits[..., 1] > roe_logits[..., 0]) & has_data_c2
        c2_tie_break = has_data_c1 & has_data_c2 & (roe_logits[..., 0] == roe_logits[..., 1]) & (c_sec < c_pred)
        vote_c2 = c2_better | c2_tie_break
        binary_votes_c2 = vote_c2.sum(dim=0)

        # The True Runoff Winner
        pred_wins = (binary_votes_c1 > binary_votes_c2) | ((binary_votes_c1 == binary_votes_c2) & (c_pred < c_sec))
        roe_prediction = torch.where(pred_wins, c_pred, c_sec)

        # Calculate the Runoff Margin (cert_r2)
        # The number of flips needed to make C2 beat C1 is (Votes_C1 - Votes_C2) // 2
        # We use // 2 because each poisoned sample change can shift the margin by 2
        cert_r2 = torch.clamp((binary_votes_c1 - binary_votes_c2 + 1) // 2, min=0)

        # If C1 actually lost the runoff (pred_wins is False), margin is 0
        cert_r2 = torch.where(pred_wins, cert_r2, torch.zeros_like(cert_r2))

        # Final Certificate: The weaker of the two rounds
        agg_margins = torch.min(cert_r1, cert_r2)

        if return_prediction:
            return agg_margins, roe_prediction
        return agg_margins
