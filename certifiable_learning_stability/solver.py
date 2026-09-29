import os
from copy import deepcopy

import gurobipy as gp
import numpy as np
import torch
from loguru import logger

from .threats import Constraints

MAX_TOPK = 1000
NOT_IN_TOPK_THRESHOLD = -30000.0


def get_gurobi_license_params() -> dict:
    root_dir = __file__.rsplit("/", 2)[0]
    license_file = os.path.join(root_dir, "gurobi.lic")
    params = {}
    with open(license_file, "r") as f:
        for line in f.readlines():
            key, val = line.strip().split("=")
            if key == "LICENSEID":
                val = int(val)
            params[key] = val

    return params


def certify_batch_targeted_attacks(
    prediction_per_partition: torch.Tensor,
    aggregation_margins: torch.Tensor,
    avoid_classes: torch.Tensor,
    partition_dset_size: int,
    reduce_classes: list[int],
    k_poison: int,
    device: torch.device,
):
    gurobi_license_params = get_gurobi_license_params()
    with gp.Env(params=gurobi_license_params) as env, gp.Model("Worst Case Adversarial Allocation Vector", env=env) as model:
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", 300)
        model.setParam("OptimalityTol", 1e-5)
        # Loosen optimality tolerance if desired
        # gp_model.setParam('MIPGap', 1e-2)  # Allow 1% gap

        num_datapoints = prediction_per_partition.shape[0]
        ensemble_size = prediction_per_partition.shape[1]
        intrinsic_rob = np.zeros((ensemble_size, num_datapoints))

        assert aggregation_margins.shape[0] == num_datapoints, "Aggregation margins tensor must have shape (num_datapoints,)"
        assert avoid_classes.shape[0] == num_datapoints, "Avoid classes tensor must have shape (num_datapoints,)"
        assert len(reduce_classes) == num_datapoints, "Reduce classes list must have dim 1 equal to num_datapoints"

        # Relaxing p to continuous for faster solving (gives the same result)
        p = model.addVars(ensemble_size, vtype=gp.GRB.CONTINUOUS, lb=0, name="poisoning_vector")  # poisoning vector that should sum up to N
        z = model.addVars(num_datapoints, vtype=gp.GRB.BINARY, name="pred_flipped_indicator")

        # Outer loop, compute outer sum: sum_k(1{g_k <= sum_i(1{p[i] > b[i][k]})})
        for k in range(num_datapoints):  # for each data point
            ### Create binary indicator variables for whether the prediction is c_s* ###
            rc_for_dpoint = set(reduce_classes[k])
            is_c_reduce_class = torch.tensor([1 if prediction_per_partition[k][i] in rc_for_dpoint else 0 for i in range(ensemble_size)]).to(
                dtype=torch.int64, device=device
            )

            # Create decision variables
            z_k = model.addVars(ensemble_size, vtype=gp.GRB.BINARY, name=f"z_{k}")  # Binary indicator variables for {p_i > b_ik}

            for i in range(ensemble_size):
                ### Populate bs ###
                if prediction_per_partition[k][i] == avoid_classes[k]:
                    intrinsic_rob[i][k] = k_poison + 1
                else:
                    intrinsic_rob[i][k] = 1

                # compute 1{p[i] >= b[i][k]}
                model.addGenConstrIndicator(z_k[i], 1, p[i] - intrinsic_rob[i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_{i}{k}")

            # compute inner sum (#reduction in votes): sum_i(1{p[i] >b[i][k]} (1 + 1{y_i=c_pred}))
            num_flipped_votes = gp.quicksum(z_k[i] * (1 + is_c_reduce_class[i]) for i in range(ensemble_size))
            # compute 1{g_k <= sum_i(1{p[i] >b[i][k]} (1 + 1{y_i=c_pred})}
            model.addGenConstrIndicator(z[k], 1, aggregation_margins[k] - num_flipped_votes, gp.GRB.LESS_EQUAL, 0, name=f"pred_flipped_indicator_{k}")

        logger.info(f"Gs: {aggregation_margins}")
        logger.info(f"Bs: {intrinsic_rob}")

        num_flipped_preds = gp.quicksum(z[i] for i in range(num_datapoints))

        # Define objective function
        model.setObjective((1 / num_datapoints) * num_flipped_preds, gp.GRB.MAXIMIZE)

        # Constraint: #total poisoned points == N
        model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)

        # Constraint: # poisoned points for each member <= batchsize
        for i in range(ensemble_size):
            model.addConstr(p[i] <= partition_dset_size)

        model.update()
        milp_stats = {
            "num_vars": model.NumVars,
            "num_constrs": model.NumConstrs,
            "num_int_vars": model.NumIntVars,
            "num_gen_constrs": model.NumGenConstrs,
        }
        model.optimize()
        milp_stats["solve_time"] = model.Runtime

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            vars = {var.VarName: var.X for var in model.getVars()}

            # Extract p and z values by matching their variable names
            p_values = [vars[f"poisoning_vector[{i}]"] for i in range(ensemble_size)]
            z_values = [vars[f"pred_flipped_indicator[{k}]"] for k in range(num_datapoints)]
            worst_case_accuracy = 1 - model.objVal
            opt_gap = model.MIPGap
            milp_stats["optimality_gap"] = opt_gap
            if model.status == gp.GRB.TIME_LIMIT:
                logger.info("Gurobi reached time limit, returning dual solution found.")
                worst_case_accuracy = 1 - model.ObjBound
            logger.info(f"Worst case flipped: {p_values}")
            logger.info(f"Worst case accuracy {worst_case_accuracy}")
            logger.info(f"Solve time: {model.Runtime:.4f} seconds")
            logger.info(f"Optimality gap: {opt_gap:.4f}")
            return worst_case_accuracy, milp_stats

        print("MILP cannot be solved.")
        return None, milp_stats


def certify_batch_dpa(
    intrinsic_rob: np.ndarray,
    prediction_per_partition: torch.Tensor,
    batch_labels: torch.Tensor,
    num_classes: int,
    partition_dset_size: int,
    k_poison: int,
    device: torch.device,
    # Allow for custom aggregation margins if needed
) -> tuple[float | None, dict]:
    intrinsic_rob = deepcopy(intrinsic_rob)
    gurobi_license_params = get_gurobi_license_params()
    with gp.Env(params=gurobi_license_params) as env, gp.Model("Worst Case Adversarial Allocation Vector", env=env) as model:
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", 300)
        model.setParam("OptimalityTol", 1e-5)
        # Loosen optimality tolerance if desired
        # gp_model.setParam('MIPGap', 1e-2)  # Allow 1% gap

        num_datapoints = batch_labels.shape[0]
        ensemble_size = prediction_per_partition.shape[1]

        agg_margins = torch.zeros(num_datapoints, dtype=torch.int64).to(device)
        # intrinsic_rob = torch.zeros((ensemble_size, num_datapoints)).to(device)

        # Relaxing p to continuous for faster solving (gives the same result)
        p = model.addVars(ensemble_size, vtype=gp.GRB.CONTINUOUS, lb=0, name="poisoning_vector")  # poisoning vector that should sum up to N
        z = model.addVars(num_datapoints, vtype=gp.GRB.BINARY, name="pred_flipped_indicator")

        # Outer loop, compute outer sum: sum_k(1{g_k <= sum_i(1{p[i] > b[i][k]})})
        for k in range(num_datapoints):  # for each data point
            ### Populate gs ###
            # Ensemble prediction
            votes_per_class = np.bincount(prediction_per_partition[k].to(torch.int64).cpu().numpy(), minlength=num_classes)

            # Sort counts in descending order
            arg_sorted_votes = np.argsort(votes_per_class, kind="stable")

            c_pred = arg_sorted_votes[-1]
            c_sec = arg_sorted_votes[-2]

            if c_pred == batch_labels[k]:
                # G = gap(c_pred, c_sec)
                agg_margins[k] = votes_per_class[c_pred] - votes_per_class[c_sec] + (c_sec > c_pred)
            else:
                agg_margins[k] = -1

            ### Create binary indicator variables for whether the prediction is c_pred ###
            is_c_pred_k = [1 if prediction_per_partition[k][i] == c_pred else 0 for i in range(ensemble_size)]
            is_c_pred_k = torch.tensor(is_c_pred_k, dtype=torch.int64).to(device)

            # Create decision variables
            z_k = model.addVars(ensemble_size, vtype=gp.GRB.BINARY, name=f"z_{k}")  # Binary indicator variables for {p_i > b_ik}

            for i in range(ensemble_size):
                ### Populate bs ###
                if prediction_per_partition[k][i] == c_sec:
                    intrinsic_rob[i][k] = k_poison + 1
                else:
                    intrinsic_rob[i][k] += 1

                # compute 1{p[i] >= b[i][k]}
                model.addGenConstrIndicator(z_k[i], 1, p[i] - intrinsic_rob[i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_{i}{k}")

            # compute inner sum (#reduction in votes): sum_i(1{p[i] >b[i][k]} (1 + 1{y_i=c_pred}))
            num_flipped_votes = gp.quicksum(z_k[i] * (1 + is_c_pred_k[i]) for i in range(ensemble_size))
            # compute 1{g_k <= sum_i(1{p[i] >b[i][k]} (1 + 1{y_i=c_pred})}
            model.addGenConstrIndicator(z[k], 1, agg_margins[k] - num_flipped_votes, gp.GRB.LESS_EQUAL, 0, name=f"pred_flipped_indicator_{k}")

        logger.info(f"Gs: {agg_margins}")
        logger.info(f"Bs: {intrinsic_rob}")

        num_flipped_preds = gp.quicksum(z[i] for i in range(num_datapoints))

        # Define objective function
        model.setObjective((1 / num_datapoints) * num_flipped_preds, gp.GRB.MAXIMIZE)

        # Constraint: #total poisoned points == N
        model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)

        # Constraint: # poisoned points for each member <= batchsize
        for i in range(ensemble_size):
            model.addConstr(p[i] <= partition_dset_size)

        model.update()
        milp_stats = {
            "num_vars": model.NumVars,
            "num_constrs": model.NumConstrs,
            "num_int_vars": model.NumIntVars,
            "num_gen_constrs": model.NumGenConstrs,
        }
        model.optimize()
        milp_stats["solve_time"] = model.Runtime

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            vars = {var.VarName: var.X for var in model.getVars()}

            # Extract p and z values by matching their variable names
            p_values = [vars[f"poisoning_vector[{i}]"] for i in range(ensemble_size)]
            z_values = [vars[f"pred_flipped_indicator[{k}]"] for k in range(num_datapoints)]
            worst_case_accuracy = 1 - model.objVal
            opt_gap = model.MIPGap
            milp_stats["optimality_gap"] = opt_gap
            if model.status == gp.GRB.TIME_LIMIT:
                logger.info("Gurobi reached time limit, returning dual solution found.")
                worst_case_accuracy = 1 - model.ObjBound
            logger.info(f"Worst case flipped: {p_values}")
            logger.info(f"Worst case accuracy {worst_case_accuracy}")
            logger.info(f"Solve time: {model.Runtime:.4f} seconds")
            logger.info(f"Optimality gap: {opt_gap:.4f}")
            return worst_case_accuracy, milp_stats

        print("MILP cannot be solved.")
        return None, milp_stats


def old_dpa_roe_prediction(votes: np.ndarray, scores: np.ndarray, num_classes: int, ensemble_size: int):
    """
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            top_3_classes: [c_pred, c_sec, c_third]
            R2: torch.Tensor with number of extra votes of c_pred over each other class in Round 2, this encodes the m in the formulation
    """

    arg_sorted_votes = np.argsort(votes, kind="stable")
    top_3_classes = torch.zeros(3, dtype=torch.int)  # [c_pred, c_sec, c_third]

    # top 3 classes
    m1 = arg_sorted_votes[-1]
    m2 = arg_sorted_votes[-2]
    m3 = arg_sorted_votes[-3]

    # DPA+ROE
    m1_election = np.zeros(num_classes, dtype=int)
    m2_election = np.zeros(num_classes, dtype=int)
    for class_idx in range(num_classes):
        # number of extra models that would vote for m1 instead of class_idx
        m1_election[class_idx] = 2 * (scores[:, m1] > scores[:, class_idx]).sum().item() - ensemble_size
        # number of extra models that would vote for m2 instead of class_idx
        m2_election[class_idx] = 2 * (scores[:, m2] > scores[:, class_idx]).sum().item() - ensemble_size

    # DPA+ROE prediction
    elec = m1_election[m2]  # number of extra models that would vote for m1 instead of m2
    # DPA+ROE prediction
    idx_dpa_roe = -1
    if elec > 0:  # m1 wins
        idx_dpa_roe = m1
    elif elec == 0:  # tie
        if m1 <= m2:
            idx_dpa_roe = m1
        else:
            idx_dpa_roe = m2
    else:  # m2 wins
        idx_dpa_roe = m2

    c_pred = idx_dpa_roe
    c_sec = m1 + m2 - c_pred  # either m1=c_pred or m2=c_pred
    c_third = m3  # top class that didn't make it to R2
    top_3_classes[0] = c_pred
    top_3_classes[1] = c_sec
    top_3_classes[2] = c_third

    # keep track of the number of extra votes of c_pred over each other class
    if c_pred == m1:
        R2 = m1_election
    else:
        R2 = m2_election

    return top_3_classes, R2


def dpa_roe_prediction(votes: np.ndarray, scores: np.ndarray, num_classes: int, ensemble_size: int):
    """
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            top_3_classes: [c_pred, c_sec, c_third]
            R2: torch.Tensor with number of extra votes of c_pred over each other class in Round 2, this encodes the m in the formulation
    """

    arg_sorted_votes = np.argsort(votes, kind="stable")
    top_3_classes = torch.zeros(3, dtype=torch.int)  # [c_pred, c_sec, c_third]

    # top 3 classes
    m1 = arg_sorted_votes[-1]
    m2 = arg_sorted_votes[-2]
    m3 = arg_sorted_votes[-3]

    def compute_margins(leader, scores_matrix):
        margins = np.zeros(num_classes, dtype=int)
        leader_scores = scores_matrix[:, leader]
        has_data_leader = leader_scores > NOT_IN_TOPK_THRESHOLD

        for c in range(num_classes):
            comp_scores = scores_matrix[:, c]
            has_data_comp = comp_scores > NOT_IN_TOPK_THRESHOLD

            # V_leader: Leader is better OR (Tied AND Leader has index priority)
            # Note: BOTH must have data to tie-break; otherwise, missing is just missing.
            leader_wins = (leader_scores > comp_scores) & has_data_leader
            leader_tie = (leader_scores == comp_scores) & has_data_leader & has_data_comp & (leader < c)
            v_leader = (leader_wins | leader_tie).sum()

            # V_comp: Competitor is better OR (Tied AND Comp has index priority)
            comp_wins = (comp_scores > leader_scores) & has_data_comp
            comp_tie = (leader_scores == comp_scores) & has_data_leader & has_data_comp & (c < leader)
            v_comp = (comp_wins | comp_tie).sum()

            # The Margin (R2 value) passed to the MILP
            margins[c] = v_leader - v_comp

        return margins

    m1_election = compute_margins(m1, scores)
    elec = m1_election[m2]  # number of extra models that would vote for m1 instead of m2

    if elec > 0:
        c_pred, c_sec, R2 = m1, m2, m1_election
    elif elec < 0:
        c_pred, c_sec = m2, m1
        R2 = compute_margins(m2, scores)  # Need m2's margins if it wins
    else:
        # Global Tie-break (m1 vs m2)
        if m1 < m2:
            c_pred, c_sec, R2 = m1, m2, m1_election
        else:
            c_pred, c_sec = m2, m1
            R2 = compute_margins(m2, scores)

    top_3_classes = torch.tensor([c_pred, c_sec, m3], dtype=torch.int)
    return top_3_classes, torch.from_numpy(R2)


def certify_batch_dpa_roe(
    intrinsic_rob: np.ndarray,
    prediction_per_partition: torch.Tensor,
    logits_per_partition: torch.Tensor,
    batch_labels: torch.Tensor,
    num_classes: int,
    partition_dset_size: int,
    k_poison: int,
    device: torch.device,
) -> tuple[float | None, dict]:
    gurobi_license_params = get_gurobi_license_params()
    with gp.Env(params=gurobi_license_params) as env, gp.Model("Worst Case Adversarial Allocation Vector", env=env) as model:
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", 300)
        model.setParam("OptimalityTol", 1e-5)
        # Loosen optimality tolerance if desired
        # gp_model.setParam('MIPGap', 1e-2)  # Allow 1% gap

        num_datapoints = len(batch_labels)
        ensemble_size = prediction_per_partition.shape[1]

        assert prediction_per_partition.shape[0] == num_datapoints, "Prediction tensor must have shape (num_datapoints, ensemble_size)"
        assert logits_per_partition.shape[0] == num_datapoints, "Logits tensor must have shape (num_datapoints, ensemble_size, num_classes)"
        assert logits_per_partition.shape[1] == ensemble_size, "Logits tensor must have shape (num_datapoints, ensemble_size, num_classes)"
        assert logits_per_partition.shape[2] == num_classes, "Logits tensor must have shape (num_datapoints, ensemble_size, num_classes)"

        bs = torch.zeros((4, ensemble_size, num_datapoints), dtype=torch.int64).to(device)
        gs = torch.zeros((4, num_datapoints), dtype=torch.int64).to(device)

        # Relaxing p to continuous for faster solving (gives the same result)
        p = model.addVars(ensemble_size, vtype=gp.GRB.CONTINUOUS, lb=0, name="poisoning_vector")  # poisoning vector that should sum up to N
        z = model.addVars(num_datapoints, vtype=gp.GRB.BINARY, name="pred_flipped_indicator")

        # Outer loop, compute outer sum: sum_k(1{g_k <= sum_i(1{p[i] > b[i][k]})})
        for k in range(num_datapoints):
            ### Populate gs ###
            # Ensemble prediction
            votes = np.bincount(prediction_per_partition[k].to(torch.int64).cpu().numpy(), minlength=num_classes)  # classes 0 to n-1

            top_3_classes, R2 = dpa_roe_prediction(votes, logits_per_partition[k], num_classes, ensemble_size)

            c_pred = top_3_classes[0].item()
            c_sec = top_3_classes[1].item()
            c_third = top_3_classes[2].item()

            # gap(c_pred, c_sec)
            gap_c_sec = votes[c_pred] - votes[c_sec] + (c_sec > c_pred)
            gap_c_sec = max(gap_c_sec, 0)
            # gap(c_pred, c_third)
            gap_c_third = votes[c_pred] - votes[c_third] + (c_third > c_pred)
            gap_c_third = max(gap_c_third, 0)
            # gap(c_sec, c_third)
            gap_c_sec_third = votes[c_sec] - votes[c_third] + (c_third > c_sec)
            gap_c_sec_third = max(gap_c_sec_third, 0)

            if c_pred == batch_labels[k]:  # correct prediction
                # G_1 = gap(c_pred, c_sec) + gap(c_pred, c_third)
                gs[0][k] = gap_c_sec + gap_c_third
                # G_2 = ceil(m(c_pred, c_sec)/2) - m= # of extra votes for c_pred over c_sec in Round 2
                gs[1][k] = np.ceil((R2[c_sec] + (c_sec > c_pred)) / 2)
                # G_3 = gap(c_sec, c_third)
                gs[2][k] = gap_c_sec_third
                # ?old --  # G_4 = ceil(m(c_pred, c_third)/2) - m= # of extra votes for c_pred over c_third in Round 2
                # ?old -- gs[3][k] = np.ceil((R2[c_third] + (c_third > c_pred)) / 2)
                # G_4 = min_c(ceil(m(c_pred, c_third)/2)) - m= # of extra votes for c_pred over c_third in Round 2
                # min of vote difference between c_pred and all other classes except c_sec
                gs[3][k] = min([np.ceil((R2[c] + (c > c_pred)) / 2) for c in range(num_classes) if c != c_pred and c != c_sec])
            else:
                gs[:, k] = -1

            ### Whether the prediction is c_pred, c_sec, or c_third ###
            is_c_k = torch.zeros((3, ensemble_size), dtype=torch.int64).to(device)
            is_c_k[0] = (prediction_per_partition[k] == c_pred).to(torch.int64)
            is_c_k[1] = (prediction_per_partition[k] == c_sec).to(torch.int64)
            is_c_k[2] = (prediction_per_partition[k] == c_third).to(torch.int64)

            ### Whether each model votes for c_sec over c_pred in Round 2 based on softmax scores
            c_sec_over_c_pred = (logits_per_partition[k, :, c_sec] > logits_per_partition[k, :, c_pred]).to(torch.int64)
            # ?old -- ### Whether each model votes for c_third over c_pred in Round 2 based on softmax scores
            # ?old -- c_third_over_c_pred = (logits_per_partition[k, :, c_third] > logits_per_partition[k, :, c_pred]).to(torch.int64)

            ### R_1 = K+1 if model predicts c_sec or c_third, 1 otherwise
            bs[0, :, k] = torch.where((is_c_k[1] == 1) | (is_c_k[2] == 1), torch.tensor(k_poison + 1, device=device), torch.tensor(1, device=device))
            ### R_2 = K+1 if model votes for c_sec over c_pred in Round 2, 1 otherwise
            bs[1, :, k] = torch.where(c_sec_over_c_pred == 1, torch.tensor(k_poison + 1, device=device), torch.tensor(1, device=device))
            ### R_3 = K+1 if model predicts c_third, 1 otherwise
            bs[2, :, k] = torch.where(is_c_k[2] == 1, torch.tensor(k_poison + 1, device=device), torch.tensor(1, device=device))
            # ?old -- ### R_4 = K+1 if model votes for c_third over c_pred
            # ?old -- bs[3, :, k] = torch.where(c_third_over_c_pred == 1, torch.tensor(k_poison + 1, device=device), torch.tensor(1, device=device))
            bs[3, :, k] = 1
            # * I think this is fine now
            bs[0, :, k] = torch.where(bs[0, :, k] == 1, intrinsic_rob[:, k] + 1, bs[0, :, k])
            bs[1, :, k] = torch.where((bs[1, :, k] == 1) & (is_c_k[0] == 1), intrinsic_rob[:, k] + 1, bs[1, :, k])
            # bs[2, :, k] = torch.where(bs[2, :, k] == 1, intrinsic_rob[:, k] + 1, bs[2, :, k])
            bs[3, :, k] = torch.where((bs[3, :, k] == 1) & (is_c_k[0] == 1), intrinsic_rob[:, k] + 1, bs[3, :, k])

            # Create decision variables
            z_k = model.addVars(4, ensemble_size, vtype=gp.GRB.BINARY, name=f"z_{k}")  # Binary indicator variables for {p_i > b_ik}
            l_k = model.addVars(3, vtype=gp.GRB.BINARY, name=f"l_{k}")  # Binary indicator variables for {L_k > 0} encoding 3 cases

            # compute flipped votes in all 3 cases
            for i in range(ensemble_size):
                # case 1: 1{p[i] >= R1[i]}
                model.addGenConstrIndicator(z_k[0, i], 1, p[i] - bs[0][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_1_{i}{k}")
                # case 2: 1{p[i] >= R2[i]}
                model.addGenConstrIndicator(z_k[1, i], 1, p[i] - bs[1][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_2_{i}{k}")
                # case 3.1: 1{p[i] >= R3[i]}
                model.addGenConstrIndicator(z_k[2, i], 1, p[i] - bs[2][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_3.1_{i}{k}")
                # case 3.2: 1{p[i] >= R4[i]}
                model.addGenConstrIndicator(z_k[3, i], 1, p[i] - bs[3][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_3.2_{i}{k}")

            # compute inner sums (reduction in gap)
            # case 1: sum_i (1{p[i] >= R1[i]} (1 + 2*1{y_i=c_pred}))
            inner_sum_1 = gp.quicksum(z_k[0, i] * (1 + 2 * is_c_k[0][i]) for i in range(ensemble_size))
            # case 2: sum_i (1{p[i] >= R2[i]})
            inner_sum_2 = gp.quicksum(z_k[1, i] for i in range(ensemble_size))
            # case 3.1: sum_i (1{p[i] >= R3[i]} (1 + 1{y_i=c_sec}))
            inner_sum_3_1 = gp.quicksum(z_k[2, i] * (1 + is_c_k[1][i]) for i in range(ensemble_size))
            # case 3.2: sum_i (1{p[i] >= R4[i]})
            inner_sum_3_2 = gp.quicksum(z_k[3, i] for i in range(ensemble_size))

            # compute indicator variable for whether each case has occured
            # case 1: 1{G1 <= inner_sum_1}
            model.addGenConstrIndicator(l_k[0], 1, gs[0][k] - inner_sum_1, gp.GRB.LESS_EQUAL, 0, name=f"case_1_indicator_{k}")

            # case 2: 1{G2 <= inner_sum_2}
            model.addGenConstrIndicator(l_k[1], 1, gs[1][k] - inner_sum_2, gp.GRB.LESS_EQUAL, 0, name=f"case_2_indicator_{k}")

            # case 3: both case 3.1 and 3.2 must occur
            l_k_3 = model.addVars(2, vtype=gp.GRB.BINARY, name=f"case_3_indicator_{k}")
            # case 3.1: 1{G3 <= inner_sum_3_1}
            model.addGenConstrIndicator(l_k_3[0], 1, gs[2][k] - inner_sum_3_1, gp.GRB.LESS_EQUAL, 0, name=f"case_3.1_indicator_{k}")
            # case 3.2: 1{G4 <= inner_sum_3_2}
            model.addGenConstrIndicator(l_k_3[1], 1, gs[3][k] - inner_sum_3_2, gp.GRB.LESS_EQUAL, 0, name=f"case_3.2_indicator_{k}")
            # combine case 3.1 and case 3.2
            model.addGenConstrIndicator(l_k[2], 1, l_k_3[0] + l_k_3[1], gp.GRB.EQUAL, 2, name=f"case_3_combined_indicator_{k}")

            # If any of these 3 cases hold, then z[k] = 1
            model.addGenConstrIndicator(z[k], 1, l_k[0] + l_k[1] + l_k[2], gp.GRB.GREATER_EQUAL, 1, name=f"pred_flipped_indicator_{k}")

        logger.info(f"Gs: {gs}")
        logger.info(f"Bs: {bs}")

        num_flipped_preds = gp.quicksum(z[i] for i in range(num_datapoints))

        # Define objective function
        model.setObjective((1 / num_datapoints) * num_flipped_preds, gp.GRB.MAXIMIZE)

        # Constraint: #total poisoned points == N
        model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)

        # Constraint: # poisoned points for each member <= batchsize
        for i in range(ensemble_size):
            model.addConstr(p[i] <= partition_dset_size)

        model.update()
        model.optimize()

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            vars = {var.VarName: var.X for var in model.getVars()}

            # Extract p and z values by matching their variable names
            p_values = [vars[f"poisoning_vector[{i}]"] for i in range(ensemble_size)]
            z_values = [vars[f"pred_flipped_indicator[{k}]"] for k in range(num_datapoints)]
            worst_case_accuracy = 1 - model.objVal
            opt_gap = 0
            opt_gap = model.MIPGap
            if model.status == gp.GRB.TIME_LIMIT:
                print("Gurobi reached time limit, returning dual solution found.")
                worst_case_accuracy = 1 - model.ObjBound
            logger.info(f"Worst case flipped: {p_values}")
            logger.info(f"Worst case accuracy {worst_case_accuracy}")
            logger.info(f"Solve time: {model.Runtime:.4f} seconds")
            logger.info(f"Optimality gap: {opt_gap:.4f}")
            return worst_case_accuracy

        print("MILP cannot be solved.")
        return None


def numpy_certify_batch_roe(
    intrinsic_rob: np.ndarray,
    prediction_per_partition: np.ndarray,
    logits_per_partition: np.ndarray,
    batch_labels: np.ndarray,
    num_classes: int,
    partition_dset_size: int,
    k_poison: int,
) -> tuple[float | None, dict]:
    gurobi_license_params = get_gurobi_license_params()
    with gp.Env(params=gurobi_license_params) as env, gp.Model("Worst Case Adversarial Allocation Vector", env=env) as model:
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", 20)
        model.setParam("OptimalityTol", 1e-5)

        num_datapoints = len(batch_labels)
        ensemble_size = prediction_per_partition.shape[1]

        assert prediction_per_partition.shape[0] == num_datapoints, "Prediction array must have shape (num_datapoints, ensemble_size)"
        assert logits_per_partition.shape[0] == num_datapoints, "Logits array must have shape (num_datapoints, ensemble_size, num_classes)"
        assert logits_per_partition.shape[1] == ensemble_size, "Logits array must have shape (num_datapoints, ensemble_size, num_classes)"
        assert logits_per_partition.shape[2] == num_classes, "Logits array must have shape (num_datapoints, ensemble_size, num_classes)"

        bs = np.zeros((4, ensemble_size, num_datapoints), dtype=np.int64)
        gs = np.zeros((4, num_datapoints), dtype=np.int64)

        p = model.addVars(ensemble_size, vtype=gp.GRB.CONTINUOUS, lb=0, name="poisoning_vector")
        z = model.addVars(num_datapoints, vtype=gp.GRB.BINARY, name="pred_flipped_indicator")

        for k in range(num_datapoints):
            votes = np.bincount(prediction_per_partition[k].astype(np.int64), minlength=num_classes)

            top_3_classes, R2 = dpa_roe_prediction(votes, logits_per_partition[k], num_classes, ensemble_size)

            [c_pred, c_sec, c_third] = [int(c) for c in top_3_classes]

            gap_c_sec = max(votes[c_pred] - votes[c_sec] + (c_sec > c_pred), 0)
            gap_c_third = max(votes[c_pred] - votes[c_third] + (c_third > c_pred), 0)
            gap_c_sec_third = max(votes[c_sec] - votes[c_third] + (c_third > c_sec), 0)

            if c_pred == batch_labels[k]:
                gs[0][k] = gap_c_sec + gap_c_third
                gs[1][k] = np.ceil((R2[c_sec] + (c_sec > c_pred)) / 2)
                gs[2][k] = gap_c_sec_third
                # active_indices = list(range(num_classes))
                # if num_classes > 1000:
                #     active_mask = (logits_per_partition[k] != -30000.0).any(axis=0)
                #     active_indices = np.where(active_mask)[0]
                # gs[3][k] = min([np.ceil((R2[c] + (c > c_pred)) / 2) for c in active_indices if c != c_pred and c != c_sec])
                gs[3][k] = np.ceil((R2[c_third] + (c_third > c_pred)) / 2)
            else:
                gs[:, k] = -1

            is_c_k = np.zeros((3, ensemble_size), dtype=np.int64)
            is_c_k[0] = (prediction_per_partition[k] == c_pred).astype(np.int64)
            is_c_k[1] = (prediction_per_partition[k] == c_sec).astype(np.int64)
            is_c_k[2] = (prediction_per_partition[k] == c_third).astype(np.int64)

            c_sec_over_c_pred = (logits_per_partition[k, :, c_sec] > logits_per_partition[k, :, c_pred]).astype(np.int64)

            bs[0, :, k] = np.where((is_c_k[1] == 1) | (is_c_k[2] == 1), k_poison + 1, 1)
            bs[1, :, k] = np.where(c_sec_over_c_pred == 1, k_poison + 1, 1)
            bs[2, :, k] = np.where(is_c_k[2] == 1, k_poison + 1, 1)
            bs[3, :, k] = 1

            z_k = model.addVars(4, ensemble_size, vtype=gp.GRB.BINARY, name=f"z_{k}")
            l_k = model.addVars(3, vtype=gp.GRB.BINARY, name=f"l_{k}")

            for i in range(ensemble_size):
                model.addGenConstrIndicator(z_k[0, i], 1, p[i] - bs[0][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_1_{i}{k}")
                model.addGenConstrIndicator(z_k[1, i], 1, p[i] - bs[1][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_2_{i}{k}")
                model.addGenConstrIndicator(z_k[2, i], 1, p[i] - bs[2][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_3.1_{i}{k}")
                model.addGenConstrIndicator(z_k[3, i], 1, p[i] - bs[3][i][k], gp.GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_3.2_{i}{k}")

            inner_sum_1 = gp.quicksum(z_k[0, i] * (1 + 2 * is_c_k[0][i]) for i in range(ensemble_size))
            inner_sum_2 = gp.quicksum(z_k[1, i] for i in range(ensemble_size))
            inner_sum_3_1 = gp.quicksum(z_k[2, i] * (1 + is_c_k[1][i]) for i in range(ensemble_size))
            inner_sum_3_2 = gp.quicksum(z_k[3, i] for i in range(ensemble_size))

            model.addGenConstrIndicator(l_k[0], 1, gs[0][k] - inner_sum_1, gp.GRB.LESS_EQUAL, 0, name=f"case_1_indicator_{k}")
            model.addGenConstrIndicator(l_k[1], 1, gs[1][k] - inner_sum_2, gp.GRB.LESS_EQUAL, 0, name=f"case_2_indicator_{k}")

            l_k_3 = model.addVars(2, vtype=gp.GRB.BINARY, name=f"case_3_indicator_{k}")
            model.addGenConstrIndicator(l_k_3[0], 1, gs[2][k] - inner_sum_3_1, gp.GRB.LESS_EQUAL, 0, name=f"case_3.1_indicator_{k}")
            model.addGenConstrIndicator(l_k_3[1], 1, gs[3][k] - inner_sum_3_2, gp.GRB.LESS_EQUAL, 0, name=f"case_3.2_indicator_{k}")
            model.addGenConstrIndicator(l_k[2], 1, l_k_3[0] + l_k_3[1], gp.GRB.EQUAL, 2, name=f"case_3_combined_indicator_{k}")

            model.addGenConstrIndicator(z[k], 1, l_k[0] + l_k[1] + l_k[2], gp.GRB.GREATER_EQUAL, 1, name=f"pred_flipped_indicator_{k}")

        logger.info(f"Gs: {gs}")
        logger.info(f"Bs: {bs}")

        num_flipped_preds = gp.quicksum(z[i] for i in range(num_datapoints))
        model.setObjective((1 / num_datapoints) * num_flipped_preds, gp.GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)
        for i in range(ensemble_size):
            model.addConstr(p[i] <= partition_dset_size)

        model.update()
        milp_stats = {
            "num_vars": model.NumVars,
            "num_constrs": model.NumConstrs,
            "num_int_vars": model.NumIntVars,
            "num_gen_constrs": model.NumGenConstrs,
        }
        model.optimize()
        milp_stats["solve_time"] = model.Runtime

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            vars = {var.VarName: var.X for var in model.getVars()}
            p_values = [vars[f"poisoning_vector[{i}]"] for i in range(ensemble_size)]
            z_values = [vars[f"pred_flipped_indicator[{k}]"] for k in range(num_datapoints)]
            worst_case_accuracy = 1 - model.objVal
            opt_gap = model.MIPGap
            milp_stats["optimality_gap"] = opt_gap
            if model.status == gp.GRB.TIME_LIMIT:
                print("Gurobi reached time limit, returning dual solution found.")
                worst_case_accuracy = 1 - model.ObjBound
            logger.info(f"Worst case flipped: {p_values}")
            logger.info(f"Worst case accuracy {worst_case_accuracy}")
            logger.info(f"Solve time: {model.Runtime:.4f} seconds")
            logger.info(f"Optimality gap: {opt_gap:.4f}")
            return worst_case_accuracy, milp_stats

        print("MILP cannot be solved.")
        return None, milp_stats
