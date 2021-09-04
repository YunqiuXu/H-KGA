import numpy as np
import torch
import os
from generic import get_match_result, to_np


def evaluate_with_ground_truth_graph(env, agent, num_games, level):
    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while(True):
        if game_id >= num_games:
            break
        try:
            obs, infos = env.reset()
        except:
            print("Error evaluating level {}, game {}!!!\n\n\n".format(level, game_id))
            game_id += 1
            continue
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()
        agent.init()
        chosen_actions, prev_step_dones = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
        prev_h, prev_c = None, None
        observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        agent.get_ingredient_list(current_triplets)
        whether_new_tasks = [True for _ in range(batch_size)]
        dones = [False for _ in range(batch_size)]
        still_running_mask = []
        final_scores = []
        for step_no in range(agent.eval_max_nb_steps_per_episode):
            assert (len(whether_new_tasks) == 1), "I assume there's only one env, no parallel"
            if whether_new_tasks[0] or dones[0]:
                agent.update_task_candidate_list(current_triplets)
                task_candidate_list = agent.available_task_list
                chosen_tasks, chosen_task_indices, _, _, _ = agent.act_greedy(
                                                                            observation_strings = observation_strings, 
                                                                            triplets = current_triplets, 
                                                                            action_candidate_list = task_candidate_list,
                                                                            tasks=[None], # batch = 1
                                                                            previous_h=None, 
                                                                            previous_c=None, 
                                                                            model_type="meta")
                chosen_tasks_before_parsing = [item[idx] for item, idx in zip(agent.available_task_list, chosen_task_indices)]
                agent.update_chosen_task_type(chosen_tasks,chosen_task_indices)
            chosen_actions, chosen_action_indices, _, prev_h, prev_c = agent.act_greedy(
                                                                            observation_strings = observation_strings, 
                                                                            triplets = current_triplets, 
                                                                            action_candidate_list = action_candidate_list, 
                                                                            tasks = chosen_tasks_before_parsing, 
                                                                            previous_h = prev_h, 
                                                                            previous_c = prev_c,
                                                                            model_type = "sub")
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_action_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)
            observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
            still_running = [1.0 - float(item) for item in prev_step_dones]
            _, whether_new_tasks = agent.get_task_rewards(current_triplets)
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)
            if np.sum(still_running) == 0:
                break
        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size
    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list
    print_strings = []
    print_strings.append("EvLevel {}|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}".format(level, 
                                                                                    np.mean(achieved_game_points), 
                                                                                    np.mean(normalized_game_points), 
                                                                                    np.mean(total_game_steps)))
    # for i in range(len(game_name_list)):
    #     print_strings.append("GameID: {}|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}".format(game_name_list[i], achieved_game_points[i], normalized_game_points[i], total_game_steps[i]))
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), 0.0, print_strings


def evaluate(env, agent, num_games, level):
    assert (agent.fully_observable_graph), "Only allow full graph!"
    return evaluate_with_ground_truth_graph(env, agent, num_games, level)

#     if agent.fully_observable_graph:
#         return evaluate_with_ground_truth_graph(env, agent, num_games, level)
#     achieved_game_points = []
#     total_game_steps = []
#     game_name_list = []
#     game_max_score_list = []
#     game_id = 0
#     while(True):
#         if game_id >= num_games:
#             break
#         obs, infos = env.reset()
#         # filter look and examine actions
#         for commands_ in infos["admissible_commands"]:
#             for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
#                 commands_.remove(cmd_)
#         game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
#         game_max_score_list += [game.max_score for game in infos["game"]]
#         batch_size = len(obs)
#         agent.eval()
#         agent.init()
# 
#         triplets, chosen_actions, prev_game_facts = [], [], []
#         prev_step_dones = []
#         for _ in range(batch_size):
#             chosen_actions.append("restart")
#             prev_game_facts.append(set())
#             triplets.append([])
#             prev_step_dones.append(0.0)
#             
#         prev_h, prev_c = None, None
# 
#         observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=None)
#         observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
#         still_running_mask = []
#         final_scores = []
# 
#         for step_no in range(agent.eval_max_nb_steps_per_episode):
# 
#             # choose what to do next from candidate list
#             chosen_actions, chosen_indices, _, prev_h, prev_c = agent.act_greedy(observation_strings, current_triplets, action_candidate_list, prev_h, prev_c)
#             # send chosen actions to game engine
#             chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_indices)]
#             obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
#             # filter look and examine actions
#             for commands_ in infos["admissible_commands"]:
#                 for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
#                     commands_.remove(cmd_)
# 
#             prev_game_facts = current_game_facts
#             observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=prev_game_facts)
#             observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
# 
#             still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
#             prev_step_dones = dones
#             final_scores = scores
#             still_running_mask.append(still_running)
# 
#             # if all ended, break
#             if np.sum(still_running) == 0:
#                 break
# 
#         achieved_game_points += final_scores
#         still_running_mask = np.array(still_running_mask)
#         total_game_steps += np.sum(still_running_mask, 0).tolist()
#         game_id += batch_size
# 
#     achieved_game_points = np.array(achieved_game_points, dtype="float32")
#     game_max_score_list = np.array(game_max_score_list, dtype="float32")
#     normalized_game_points = achieved_game_points / game_max_score_list
#     print_strings = []
#     # print_strings.append("======================================================")
#     print_strings.append("EvLevel {}|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}".format(level, 
#                                                                                     np.mean(achieved_game_points), 
#                                                                                     np.mean(normalized_game_points), 
#                                                                                     np.mean(total_game_steps)))
#     # for i in range(len(game_name_list)):
#     #     print_strings.append("GameID: {}|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}".format(game_name_list[i], achieved_game_points[i], normalized_game_points[i], total_game_steps[i]))
#     # print_strings.append("======================================================")
#     print_strings = "\n".join(print_strings)
#     print(print_strings)
#     return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), 0.0, print_strings
