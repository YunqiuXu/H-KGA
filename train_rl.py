import time
import os
import time
import copy
import json
import tempfile
import numpy as np
from os.path import join as pjoin
from distutils.dir_util import copy_tree
from agent import Agent
import generic
from generic import HistoryScoreCache, EpisodicCountingMemory
import evaluate
import reinforcement_learning_dataset

def np_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def train():
    time_1 = time.time()
    print("Load config")
    config = generic.load_config()
    output_dir = config["general"]["output_dir"]
    pretrained_dir = config["general"]["pretrained_dir"]
    games_dir = config["general"]["games_dir"]
    
    print("===== ----- ===== Initialize Agent ===== ----- =====")
    agent = Agent(config)
    requested_infos = agent.select_additional_infos()
    
    print("===== ----- ===== Build training envs, L3/5/7/9, 100 games per ===== ----- =====")
    env_dict = {}
    for difficulty_level in [3,7,5,9]:
        env, _ = reinforcement_learning_dataset.get_training_game_env_1level100game(
                                                            data_dir = games_dir+config['rl']['data_path'],
                                                            difficulty_level = difficulty_level,
                                                            requested_infos = requested_infos,
                                                            max_episode_steps = agent.max_nb_steps_per_episode,
                                                            batch_size = agent.batch_size)
        env_dict[difficulty_level] = env
    print("Training levels: {}".format(list(env_dict.keys())))

    if agent.run_eval:
        print("===== ----- ===== Build evaluating (validating) envs, 4 levels, 20 games per ===== ----- =====")
        eval_env_dict = {}
        for difficulty_level in [3, 7, 5, 9]:
            eval_title = "eval_level_{}".format(difficulty_level)
            eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(games_dir+config['rl']['data_path'],
                                                                                         difficulty_level,
                                                                                         requested_infos,
                                                                                         agent.eval_max_nb_steps_per_episode,
                                                                                         agent.eval_batch_size,
                                                                                         valid_or_test="valid")    
            eval_env_dict[eval_title] = {"eval_env": eval_env, "num_eval_game": num_eval_game}
            print("{}, {} games".format(eval_title, eval_env_dict[eval_title]["num_eval_game"]))
            print("-----")

        print("===== ----- ===== Build testing envs, 8 levels, 20 games per ===== ----- =====")
        test_env_dict = {}
        for difficulty_level in [1,2,3,4,5,7,8,9]:
            test_title = "test_level_{}".format(difficulty_level)
            test_env, num_test_game = reinforcement_learning_dataset.get_evaluation_game_env(games_dir+config['rl']['data_path'],
                                                                                         difficulty_level,
                                                                                         requested_infos,
                                                                                         agent.eval_max_nb_steps_per_episode,
                                                                                         agent.eval_batch_size,
                                                                                         valid_or_test="test")    
            test_env_dict[test_title] = {"test_env": test_env, "num_test_game": num_test_game}
            print("{}, {} games".format(test_title, test_env_dict[test_title]["num_test_game"]))
            print("-----")
    else:
        raise Exception("No validating and evaluating envs")
        # eval_env, num_eval_game = None, None
        # test_env, num_test_game = None, None

    # disable visdom
    assert (not config["general"]["visdom"]), "Disable visdom"
    
    # Load pretrained weights if needed (not used)
    if os.path.exists(pretrained_dir + "/" + agent.load_graph_update_model_from_tag + ".pt"):
        print("Load pretrained graph_update_model")
        agent.load_pretrained_command_generation_model(pretrained_dir + "/" + agent.load_graph_update_model_from_tag + ".pt")
    else:
        print("No pretrained graph_update_model")
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            print("Load checkpoint from {}".format(output_dir))
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
            agent.update_target_net()
        elif os.path.exists(pretrained_dir + "/" + agent.load_from_tag + ".pt"):
            print("Load checkpoint from {}".format(pretrained_dir))
            agent.load_pretrained_model(pretrained_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()
    else:
        print("Train GATA from scratch!")

    print("===== ----- ===== Initialize loggers ===== ----- =====")
    running_avg_game_points = HistoryScoreCache(capacity=500)             
    running_avg_game_points_normalized = HistoryScoreCache(capacity=500)  
    running_avg_game_steps = HistoryScoreCache(capacity=500)             
    running_avg_dqn_loss_meta = HistoryScoreCache(capacity=500)
    running_avg_dqn_loss_sub = HistoryScoreCache(capacity=500)    
    running_avg_game_rewards_meta = HistoryScoreCache(capacity=500)  # meta: should be same with env rewards
    running_avg_game_rewards_sub = HistoryScoreCache(capacity=500)   # sub: task_rewards
    running_avg_task_succ_rate = HistoryScoreCache(capacity=500)     # sub: episodic_task_rewards sum / number of tasks
    running_avg_graph_rewards_sub = HistoryScoreCache(capacity=500)  # sub: KGcount

    json_file_name = agent.experiment_tag.replace(" ", "_")
    print("Progress will be saved at {}/{}.json".format(output_dir, json_file_name))
    i_have_seen_these_states = EpisodicCountingMemory()  # episodic counting based memory
    step_in_total = 0
    episode_no = 0
    best_train_performance_so_far = 0.0
    best_eval_performance_so_far = 0.0
    prev_performance = 0.0
    i_am_patient = 0
    perfect_training = 0
    level_count_dict = {3:0,7:0,5:0,9:0}
    level_normalized_collection_dict = {3:[], 7:[], 5:[], 9:[]}
    level_normalized_avg_dict = {3:0.0, 7:0.0, 5:0.0, 9:0.0}   

    print("===== ----- Start training! ----- =====")
    while(True):
        if episode_no > agent.max_episode:
            break
        # sample a level
        np.random.seed(episode_no)
        level_sampling_prob = np_softmax(1.0 - np.array(list(level_normalized_avg_dict.values())))
        curr_difficulty_level = np.random.choice([3,7,5,9], p = level_sampling_prob)
        env = env_dict[curr_difficulty_level]
        level_count_dict[curr_difficulty_level] += 1
        # sample an env (shuffled)
        try: 
            env.seed(episode_no)
            obs, infos = env.reset()   
        except:  
            # sometimes there might be unexpected error
            error_log = "Error at ep{}".format(episode_no)
            print(error_log)
            episode_no += 1
            with open(output_dir + "/error_log.txt", "a") as f:
                f.write(error_log + "\n")
            continue

        agent.train()
        agent.init()

        # basic infos for this game
        game_name_list = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list = [game.max_score for game in infos["game"]]
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        
        # reset episodic counting based memory
        i_have_seen_these_states.reset()  
        
        # initializa some inputs
        prev_triplets = []
        chosen_actions = []
        prev_game_facts = []
        prev_step_dones = []
        prev_rewards = []
        batch_size = len(obs)
        assert (batch_size == 1), "I assume there's only one env, no parallel"
        for _ in range(batch_size):
            prev_triplets.append([])
            chosen_actions.append("restart")
            prev_game_facts.append(set())
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)
        
        # process obs
        prev_h, prev_c = None, None
        observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=None)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        
        # update init triplets into episodic memory
        i_have_seen_these_states.push(current_triplets)  
        
        # for count reward bonus
        agent.accumulative_counting_memory.push(current_triplets)
        
        # At the beginning of each episode
        # 1. obtain ingredient_list from kg_0
        # 2. initialize available_task_list / available_task_type_list / curr_chosen_tasks / curr_chosen_task_types
        agent.get_ingredient_list(current_triplets)
        whether_new_tasks = [True for _ in range(batch_size)]
        dones = [False for _ in range(batch_size)]
        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode
        game_points = []                    
        print_actions = []                 
        print_tasks = []                   

        # Collect the transition separately
        transition_cache_meta = []
        transition_cache_sub = []

        # Check the dones separately
        still_running_mask_meta = []
        still_running_mask_sub = []

        # For sub: collect game rewards and graph rewards
        graph_rewards_sub = []
        game_rewards_sub = []

        # For meta: collect game rewards
        game_rewards_meta = []

        for step_no in range(agent.max_nb_steps_per_episode):
            # If one True in whether_new_tasks, then re-select a task
            assert (len(whether_new_tasks) == 1), "I assume there's only one env, no parallel"
            if whether_new_tasks[0] or dones[0]:
                # 1) update the task candidate list
                agent.update_task_candidate_list(current_triplets)
                task_candidate_list = agent.available_task_list
                # 2) select a task: ignore prev_h / prev_c --> no recurrent
                chosen_tasks, chosen_task_indices, replay_info_meta, _, _ = agent.act(
                                                                            observation_strings = observation_strings, 
                                                                            triplets = current_triplets, 
                                                                            action_candidate_list = task_candidate_list,
                                                                            tasks=[None], # batch = 1
                                                                            previous_h=None, 
                                                                            previous_c=None, 
                                                                            random=act_randomly,
                                                                            model_type="meta")
                # 3) replay_info_meta: [observation_strings, current_triplets, task_candidate_list, [None], chosen_task_indices]
                transition_cache_meta.append(replay_info_meta) 
                # 4) reinitialize rewards_meta after re-selecting a task
                rewards_meta = [0. for _ in range(batch_size)]
                # 5) make selected tasks to be used for the sub model
                chosen_tasks_before_parsing = [item[idx] for item, idx in zip(agent.available_task_list, chosen_task_indices)]
                # 6) update chosen tasks / status
                agent.update_chosen_task_type(chosen_tasks,chosen_task_indices)

            # Select an action conditioned on the chosen_tasks_before_parsing
            chosen_actions, chosen_action_indices, replay_info_sub, prev_h, prev_c = agent.act(
                                                                            observation_strings = observation_strings, 
                                                                            triplets = current_triplets, 
                                                                            action_candidate_list = action_candidate_list, 
                                                                            tasks=chosen_tasks_before_parsing, 
                                                                            previous_h=prev_h, 
                                                                            previous_c=prev_c, 
                                                                            random=act_randomly,
                                                                            model_type="sub")
            # replay_info_sub: [observation_strings, current_triplets, action_candidate_list, chosen_tasks_before_parsing, chosen_action_indices]
            transition_cache_sub.append(replay_info_sub)
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_action_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)
            prev_triplets = current_triplets
            prev_game_facts = current_game_facts
            # re-generate information (as next state)
            observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=prev_game_facts)
            # observation_for_counting = copy.copy(observation_strings) # disable
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
            # episodic counting memory
            has_not_seen = i_have_seen_these_states.has_not_seen(current_triplets)
            i_have_seen_these_states.push(current_triplets)  
            # accumulative counting memory
            agent.accumulative_counting_memory.push(current_triplets)
            
            # An update step
            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                dqn_loss_meta, _ = agent.update_dqn(episode_no, model_type="meta")
                if dqn_loss_meta is not None:
                    running_avg_dqn_loss_meta.push(dqn_loss_meta)
                dqn_loss_sub, _ = agent.update_dqn(episode_no, model_type="sub")
                if dqn_loss_sub is not None:
                    running_avg_dqn_loss_sub.push(dqn_loss_sub)

            # If reach maximum steps --> set all dones as True (terminate the game)
            if step_no == agent.max_nb_steps_per_episode - 1:
                dones = [True for _ in dones]
            step_in_total += 1                                                                      # Only for determining updating
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # The r_t+1 from env
            game_points.append(copy.copy(step_rewards))                                             # For visualization
            prev_rewards = scores                                                                   # Score_t+1 = r_0 + r_1 + ... + r_t+1
            still_running = [1.0 - float(item) for item in prev_step_dones]                         # DQN requires one extra step
            prev_step_dones = dones                                          
            
            # graph rewards: for sub
            step_graph_rewards = agent.get_graph_rewards(prev_triplets, current_triplets)  # list of float
            if agent.graph_reward_type != "KGcount_acc": # "triplets_increased", "triplets_difference", "KGcount_epi", "KGcount_bebold"
                step_graph_rewards = [r * float(m) for r, m in zip (step_graph_rewards, has_not_seen)]
            graph_rewards_sub.append(step_graph_rewards)
            
            # task rewards: for sub, check whether a task has done
            task_rewards, whether_new_tasks = agent.get_task_rewards(current_triplets)
            game_rewards_sub.append(task_rewards)
            
            # step rewards: for meta, keep accumulating until a task has been finished
            rewards_meta = [rm+sr for rm,sr in zip (rewards_meta, step_rewards)]
            if whether_new_tasks[0] or dones[0]:
                game_rewards_meta.append(rewards_meta)

            # update still running mask
            # For meta, the transition number should be equal to the number of tasks within this episode
            still_running_mask_sub.append(still_running)
            if whether_new_tasks[0] or dones[0]:
                still_running_mask_meta.append(still_running)

            # Add some printing informations for each timestep
            print_actions.append(chosen_actions_before_parsing[0] if still_running[0] else "--")
            print_tasks.append(chosen_tasks_before_parsing[0] if still_running[0] else "--")
            
            if np.sum(still_running) == 0: # if all ended, break
                break

        # After an episode
        # 1) The sub model
        # 1a) mask
        still_running_mask_sub_np = np.array(still_running_mask_sub)                    # steps within episode x batch
        # 1b) game rewards
        game_rewards_sub_np = np.array(game_rewards_sub) * still_running_mask_sub_np    # steps within episode x batch
        game_rewards_sub_pt = generic.to_pt(game_rewards_sub_np, enable_cuda=agent.use_cuda, type='float')
        # 1c) graph rewards
        graph_rewards_sub_np = np.array(graph_rewards_sub) * still_running_mask_sub_np  # steps within episode x batch
        if agent.graph_reward_lambda > 0.0:
            graph_rewards_sub_pt = generic.to_pt(graph_rewards_sub_np, enable_cuda=agent.use_cuda, type='float')  
        else:
            graph_rewards_sub_pt = generic.to_pt(np.zeros_like(graph_rewards_sub_np), enable_cuda=agent.use_cuda, type='float') 
        # 1d) count rewards: same shape with game rewards
        count_rewards_sub_pt = generic.to_pt(np.zeros_like(game_rewards_sub_np), enable_cuda=agent.use_cuda, type='float')
        # 2) The meta model
        # 2a) mask
        still_running_mask_meta_np = np.array(still_running_mask_meta)                  # tasks within episode x batch
        # 2b) game rewards
        game_rewards_meta_np = np.array(game_rewards_meta) * still_running_mask_meta_np # steps within episode x batch
        game_rewards_meta_pt = generic.to_pt(game_rewards_meta_np, enable_cuda=agent.use_cuda, type='float')
        # 2c) graph rewards: same shape with game rewards
        graph_rewards_meta_pt = generic.to_pt(np.zeros_like(game_rewards_meta_np), enable_cuda=agent.use_cuda, type='float')
        # 2d) count rewards: same shape with game rewards
        count_rewards_meta_pt = generic.to_pt(np.zeros_like(game_rewards_meta_np), enable_cuda=agent.use_cuda, type='float')
        # 3) All
        game_points_np = np.array(game_points) * still_running_mask_sub_np              # steps within episode x batch

        # Push experience into replay buffer (dqn, disable drqn)
        # 1) The sub memory
        avg_rewards_in_buffer_sub = agent.sub_dqn_memory.avg_rewards_dict[curr_difficulty_level]
        for b in range(game_rewards_sub_np.shape[1]):
            if still_running_mask_sub_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_sub_np[-1][b] != 0:
                _need_pad_sub = True # need to pad one transition
                tmp_game_rewards_sub = game_rewards_sub_np[:, b].tolist() + [0.0]
            else:
                _need_pad_sub = False
                tmp_game_rewards_sub = game_rewards_sub_np[:, b]
            # Do not push if reward of this episode is smaller than in buffer
            if np.mean(tmp_game_rewards_sub) < avg_rewards_in_buffer_sub * agent.buffer_reward_threshold:
                continue
            for i in range(game_rewards_sub_np.shape[0]):
                observation_strings, _triplets, action_candidate_list, tasks, chosen_action_indices = transition_cache_sub[i]
                is_final_sub = True
                if still_running_mask_sub_np[i][b] != 0:
                    is_final_sub = False
                agent.sub_dqn_memory.add(observation_strings[b], 
                                         action_candidate_list[b], 
                                         tasks[b], 
                                         chosen_action_indices[b], 
                                         _triplets[b], 
                                         game_rewards_sub_pt[i][b], 
                                         graph_rewards_sub_pt[i][b], 
                                         count_rewards_sub_pt[i][b], 
                                         is_final_sub,
                                         curr_difficulty_level)
                if still_running_mask_sub_np[i][b] == 0:
                    break
            if _need_pad_sub:
                observation_strings, _triplets, action_candidate_list, tasks, chosen_action_indices = transition_cache_sub[-1]
                agent.sub_dqn_memory.add(observation_strings[b], 
                                         action_candidate_list[b], 
                                         tasks[b],
                                         chosen_action_indices[b], 
                                         _triplets[b], 
                                         game_rewards_sub_pt[-1][b] * 0.0, 
                                         graph_rewards_sub_pt[-1][b] * 0.0, 
                                         count_rewards_sub_pt[-1][b] * 0.0, 
                                         True,
                                         curr_difficulty_level)
        agent.sub_dqn_memory.update_avg_rewards()

        # 2) The meta memory
        avg_rewards_in_buffer_meta = agent.meta_dqn_memory.avg_rewards_dict[curr_difficulty_level]
        for b in range(game_rewards_meta_np.shape[1]):
            if still_running_mask_sub_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_meta_np[-1][b] != 0:
                _need_pad_meta = True # need to pad one transition
                tmp_game_rewards_meta = game_rewards_meta_np[:, b].tolist() + [0.0]
            else:
                _need_pad_meta = False
                tmp_game_rewards_meta = game_rewards_meta_np[:, b]
            # Do not push if reward of this episode is smaller than in buffer
            if np.mean(tmp_game_rewards_meta) < avg_rewards_in_buffer_meta * agent.buffer_reward_threshold:
                continue
            for i in range(game_rewards_meta_np.shape[0]):
                observation_strings, _triplets, task_candidate_list, tasks, chosen_task_indices = transition_cache_meta[i]
                is_final_meta = True
                if still_running_mask_meta_np[i][b] != 0:
                    is_final_meta = False
                assert (tasks[b] is None), "meta memory: tasks[b] should be None, got {}".format(tasks[b])
                agent.meta_dqn_memory.add(observation_strings[b], 
                                         task_candidate_list[b], 
                                         tasks[b], 
                                         chosen_task_indices[b], 
                                         _triplets[b], 
                                         game_rewards_meta_pt[i][b], 
                                         graph_rewards_meta_pt[i][b], 
                                         count_rewards_meta_pt[i][b], 
                                         is_final_meta,
                                         curr_difficulty_level) 
                if still_running_mask_meta_np[i][b] == 0:
                    break
            if _need_pad_meta:
                observation_strings, _triplets, task_candidate_list, tasks, chosen_task_indices = transition_cache_meta[-1]
                assert (tasks[b] is None), "meta memory: tasks[b] should be None, got {}".format(tasks[b])
                agent.meta_dqn_memory.add(observation_strings[b], 
                                         task_candidate_list[b], 
                                         tasks[b],
                                         chosen_task_indices[b], 
                                         _triplets[b], 
                                         game_rewards_meta_pt[-1][b] * 0.0, 
                                         graph_rewards_meta_pt[-1][b] * 0.0, 
                                         count_rewards_meta_pt[-1][b] * 0.0, 
                                         True,
                                         curr_difficulty_level) 
        agent.meta_dqn_memory.update_avg_rewards()

        # Log training progress
        for b in range(batch_size):
            # All
            running_avg_game_points.push(np.sum(game_points_np, 0)[b])
            game_max_score_np = np.array(game_max_score_list, dtype="float32")
            running_avg_game_points_normalized.push((np.sum(game_points_np, 0) / game_max_score_np)[b])
            running_avg_game_steps.push(np.sum(still_running_mask_sub_np, 0)[b])
            # Sub
            running_avg_game_rewards_sub.push(np.sum(game_rewards_sub_np, 0)[b])
            # Assumee that batch == 1
            running_avg_task_succ_rate.push(np.sum(game_rewards_sub_np, 0)[b] / len(still_running_mask_meta))
            running_avg_graph_rewards_sub.push(np.sum(graph_rewards_sub_np, 0)[b])
            # Meta
            running_avg_game_rewards_meta.push(np.sum(game_rewards_meta_np, 0)[b])
            level_normalized_collection_dict[curr_difficulty_level].append((np.sum(game_points_np, 0) / game_max_score_np)[b])
            level_normalized_avg_dict[curr_difficulty_level] = np.mean(level_normalized_collection_dict[curr_difficulty_level])

        # Finish game, if not reach "start learning" or report freq, do not print progress / test / save model
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size
        if episode_no < agent.learn_start_from_this_episode:
            if episode_no % 100 == 0:
                print("----- Epi {} ----- \nLevel count {}\nNorm avg {}\nSampling prob {}".format(
                    episode_no, level_count_dict, level_normalized_avg_dict, level_sampling_prob))
            continue
        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - batch_size) % agent.report_frequency):
            if episode_no % 100 == 0:
                print("----- Epi {} ----- \nLevel count {}\nNorm avg {}\nSampling prob {}".format(
                    episode_no, level_count_dict, level_normalized_avg_dict, level_sampling_prob))
            continue
 
        # Print training progress
        time_2 = time.time()
        progress_train_p1 = "Train|Epi: {:3d}|Time: {:.2f}m|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}|"
        progress_train_p1 = progress_train_p1.format(
                            episode_no, 
                            (time_2 - time_1) / 60., 
                            running_avg_game_points.get_avg(), 
                            running_avg_game_points_normalized.get_avg(), 
                            running_avg_game_steps.get_avg())
        progress_train_p2 = "L_DQN_meta: {:2.3f}|L_DQN_sub: {:2.3f}|"
        progress_train_p2 = progress_train_p2.format(
                            running_avg_dqn_loss_meta.get_avg(), 
                            running_avg_dqn_loss_sub.get_avg())
        progress_train_p3 = "Rew_meta: {:2.3f}|Rew_sub: {:2.3f}|RewGraph_sub: {:2.3f}|EpiTaskSuccRate: {:2.3f}|"
        progress_train_p3 = progress_train_p3.format(
                            running_avg_game_rewards_meta.get_avg(), 
                            running_avg_game_rewards_sub.get_avg(),
                            running_avg_graph_rewards_sub.get_avg(), 
                            running_avg_task_succ_rate.get_avg())
        progress_train_p4 = "L3 C {}, NA {:2.3f}, SP {:2.3f}|L7 C {}, NA {:2.3f}, SP {:2.3f}|L5 C {}, NA {:2.3f}, SP {:2.3f}|L9 C {}, NA {:2.3f}, SP {:2.3f}"
        progress_train_p4 = progress_train_p4.format(
                                level_count_dict[3], level_normalized_avg_dict[3], level_sampling_prob[0],
                                level_count_dict[7], level_normalized_avg_dict[7], level_sampling_prob[1],
                                level_count_dict[5], level_normalized_avg_dict[5], level_sampling_prob[2],
                                level_count_dict[9], level_normalized_avg_dict[9], level_sampling_prob[3],
                            )
        print(progress_train_p1)
        print(progress_train_p2)
        print(progress_train_p3)
        # print(progress_train_p4)
        progress_train_to_write = progress_train_p1 + progress_train_p2 + progress_train_p3 + progress_train_p4
        with open(output_dir + "/training_log.txt", 'a') as f:
            f.write(progress_train_to_write + "\n")

        print(game_name_list[0] + ":    " + " | ".join(print_actions))
        print("\nIngredients: {}".format(agent.ingredient_list_batch[0]))
        print("\nTasks per timestep: ")
        print(print_tasks)
        print("----- ----- ----- -----")

        print("====================================================== Eval start")
        curr_train_performance = running_avg_game_points_normalized.get_avg()
        assert (agent.run_eval), "agent.run_eval should be True!"
        eval_performance_dict = {}
        eval_game_points_normalized_list = []

        # Validating
        print("=============== Validating")
        for difficulty_level in [3,7,5,9]:
            eval_title = "eval_level_{}".format(difficulty_level)
            eval_env = eval_env_dict[eval_title]["eval_env"]
            num_eval_game = eval_env_dict[eval_title]["num_eval_game"]
            eval_game_points, eval_game_points_normalized, eval_game_step, _, detailed_scores = evaluate.evaluate(
                                                                                                            eval_env, 
                                                                                                            agent, 
                                                                                                            num_eval_game,
                                                                                                            difficulty_level)
            eval_performance_dict[eval_title] = {"eval_game_points":eval_game_points, 
                                                "eval_game_points_normalized":eval_game_points_normalized,
                                                "eval_game_step":eval_game_step,
                                                "detailed_scores":detailed_scores}
            assert (difficulty_level in {3,7,5,9}), "For evaluating (validating), difficulty_level should be in {3,7,5,9}, got {}".format(difficulty_level)
            eval_game_points_normalized_list.append(eval_game_points_normalized)
        
        # Testing
        print("=============== Testing")
        test_performance_dict = {}
        for difficulty_level in [1,2,3,4,5,7,8,9]:
            test_title = "test_level_{}".format(difficulty_level)
            test_env = test_env_dict[test_title]["test_env"]
            num_test_game = test_env_dict[test_title]["num_test_game"]
            test_game_points, test_game_points_normalized, test_game_step, _, test_detailed_scores = evaluate.evaluate(
                                                                                                                test_env, 
                                                                                                                agent, 
                                                                                                                num_test_game,
                                                                                                                difficulty_level)
            test_performance_dict[test_title] = {"test_game_points":test_game_points, 
                                                 "test_game_points_normalized":test_game_points_normalized,
                                                 "test_game_step":test_game_step,
                                                 "test_detailed_scores":test_detailed_scores}

        # Compute the average validate performance (for saving / reloading the model)
        curr_eval_performance = np.mean(eval_game_points_normalized_list)
        curr_performance = curr_eval_performance
        # Check whether to save model
        save_tag_total = "{}/{}_model.pt".format(output_dir, agent.experiment_tag)
        save_tag_partial = "{}/{}_model_{}00kEpi.pt".format(output_dir, agent.experiment_tag, int(episode_no // 100000 + 1))
        if curr_eval_performance > best_eval_performance_so_far:
            best_eval_performance_so_far = curr_eval_performance
            agent.save_model_to_path(save_tag_total)
            agent.save_model_to_path(save_tag_partial)
            print(save_tag_total)
            print(save_tag_partial)
        elif curr_eval_performance == best_eval_performance_so_far:
            if curr_eval_performance > 0.0:
                agent.save_model_to_path(save_tag_total)
                agent.save_model_to_path(save_tag_partial)
                print(save_tag_total)
                print(save_tag_partial)
            else:
                if curr_train_performance >= best_train_performance_so_far:
                    agent.save_model_to_path(save_tag_total)   
                    agent.save_model_to_path(save_tag_partial)
                    print(save_tag_total)
                    print(save_tag_partial)
        # update best train performance
        if curr_train_performance >= best_train_performance_so_far:
            best_train_performance_so_far = curr_train_performance
        # if patient >= patience, resume from checkpoint
        if prev_performance <= curr_performance:
            i_am_patient = 0
        else:
            i_am_patient += 1
        prev_performance = curr_performance
        if agent.patience > 0 and i_am_patient >= agent.patience:
            if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
                print('Patience exceeded! Reload from a good checkpoint...')
                print(save_tag_total)
                agent.load_pretrained_model(save_tag_total, load_partial_graph=False)
                agent.update_target_net()
                i_am_patient = 0
        if running_avg_game_points_normalized.get_avg() >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0
        print("====================================================== Eval done!")

        # write accuracies down into file
        print("====================================================== Writing")
        _s = json.dumps({
                         "Time": "{:.2f}".format((time_2 - time_1) / 60.), 
                         "TrScore": str(running_avg_game_points.get_avg()),
                         "TrScoreNorm": str(running_avg_game_points_normalized.get_avg()),
                         "TrSteps": str(running_avg_game_steps.get_avg()),
                         "L_DQN_meta": str(running_avg_dqn_loss_meta.get_avg()),
                         "L_DQN_sub": str(running_avg_dqn_loss_sub.get_avg()),
                         "TrRew_meta": str(running_avg_game_rewards_meta.get_avg()),
                         "TrRew_sub": str(running_avg_game_rewards_sub.get_avg()),
                         "TrRewGraph_sub": str(running_avg_graph_rewards_sub.get_avg()),
                         "TrEpiTaskSuccRate": str(running_avg_task_succ_rate.get_avg()), 
                         # Validating  
                         "EvScoreL3": str(eval_performance_dict["eval_level_3"]["eval_game_points"]),
                         "EvScoreNormL3": str(eval_performance_dict["eval_level_3"]["eval_game_points_normalized"]),
                         "EvStepsL3": str(eval_performance_dict["eval_level_3"]["eval_game_step"]),
                         "EvScoreL5": str(eval_performance_dict["eval_level_5"]["eval_game_points"]),
                         "EvScoreNormL5": str(eval_performance_dict["eval_level_5"]["eval_game_points_normalized"]),
                         "EvStepsL5": str(eval_performance_dict["eval_level_5"]["eval_game_step"]),         
                         "EvScoreL7": str(eval_performance_dict["eval_level_7"]["eval_game_points"]),
                         "EvScoreNormL7": str(eval_performance_dict["eval_level_7"]["eval_game_points_normalized"]),
                         "EvStepsL7": str(eval_performance_dict["eval_level_7"]["eval_game_step"]),
                         "EvScoreL9": str(eval_performance_dict["eval_level_9"]["eval_game_points"]),
                         "EvScoreNormL9": str(eval_performance_dict["eval_level_9"]["eval_game_points_normalized"]),
                         "EvStepsL9": str(eval_performance_dict["eval_level_9"]["eval_game_step"]),
                         # Testing
                         "TeScoreL1": str(test_performance_dict["test_level_1"]["test_game_points"]),
                         "TeScoreNormL1": str(test_performance_dict["test_level_1"]["test_game_points_normalized"]),
                         "TeStepsL1": str(test_performance_dict["test_level_1"]["test_game_step"]),
                         "TeScoreL2": str(test_performance_dict["test_level_2"]["test_game_points"]),
                         "TeScoreNormL2": str(test_performance_dict["test_level_2"]["test_game_points_normalized"]),
                         "TeStepsL2": str(test_performance_dict["test_level_2"]["test_game_step"]),
                         "TeScoreL3": str(test_performance_dict["test_level_3"]["test_game_points"]),
                         "TeScoreNormL3": str(test_performance_dict["test_level_3"]["test_game_points_normalized"]),
                         "TeStepsL3": str(test_performance_dict["test_level_3"]["test_game_step"]),
                         "TeScoreL4": str(test_performance_dict["test_level_4"]["test_game_points"]),
                         "TeScoreNormL4": str(test_performance_dict["test_level_4"]["test_game_points_normalized"]),
                         "TeStepsL4": str(test_performance_dict["test_level_4"]["test_game_step"]),
                         "TeScoreL5": str(test_performance_dict["test_level_5"]["test_game_points"]),
                         "TeScoreNormL5": str(test_performance_dict["test_level_5"]["test_game_points_normalized"]),
                         "TeStepsL5": str(test_performance_dict["test_level_5"]["test_game_step"]),
                         "TeScoreL7": str(test_performance_dict["test_level_7"]["test_game_points"]),
                         "TeScoreNormL7": str(test_performance_dict["test_level_7"]["test_game_points_normalized"]),
                         "TeStepsL7": str(test_performance_dict["test_level_7"]["test_game_step"]),
                         "TeScoreL8": str(test_performance_dict["test_level_8"]["test_game_points"]),
                         "TeScoreNormL8": str(test_performance_dict["test_level_8"]["test_game_points_normalized"]),
                         "TeStepsL8": str(test_performance_dict["test_level_8"]["test_game_step"]),
                         "TeScoreL9": str(test_performance_dict["test_level_9"]["test_game_points"]),
                         "TeScoreNormL9": str(test_performance_dict["test_level_9"]["test_game_points_normalized"]),
                         "TeStepsL9": str(test_performance_dict["test_level_9"]["test_game_step"]),
                        })
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()
        print("====================================================== Writing done")
        
        # break if test == 1.0 or train >= 0.95
        if curr_performance == 1.0 and curr_train_performance >= 0.95:
            break
        if perfect_training >= 3:
            break


if __name__ == '__main__':
    train()
