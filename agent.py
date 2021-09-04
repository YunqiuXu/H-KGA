
import os
import random
import copy
import codecs
import spacy
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as F
from textworld import EnvInfos

from generic import to_np, to_pt, _words_to_ids, _word_to_id, pad_sequences, update_graph_triplets, preproc, max_len, ez_gather_dim_1
from generic import sort_target_commands, process_facts, serialize_facts, gen_graph_commands, process_fully_obs_facts
from generic import generate_labels_for_ap, generate_labels_for_sp, LinearSchedule
from generic import AccumulativeCountingMemory                               

import kg_utils 
from layers import NegativeLogLoss, compute_mask, masked_mean
import dqn_memory_priortized_replay_buffer
from model_meta import TaskSelector                                          
from model_sub import KG_Manipulation                                        


class Agent:
    def __init__(self, config):
        # 1. Load config
        self.config = config
        self.load_config()

        # 2. Build model
        self.meta_net = TaskSelector(config=self.config, 
                                        word_vocab=self.word_vocab, 
                                        node_vocab=self.node_vocab, 
                                        relation_vocab=self.relation_vocab)
        self.sub_net = KG_Manipulation(config=self.config, 
                                        word_vocab=self.word_vocab, 
                                        node_vocab=self.node_vocab, 
                                        relation_vocab=self.relation_vocab)
        self.train() # set mode, and train for meta_net and sub_net
        if self.use_cuda:
            self.meta_net.cuda()
            self.sub_net.cuda()

        # 3. Set target network
        if self.task == "rl":
            self.meta_target_net = TaskSelector(config=self.config, 
                                              word_vocab=self.word_vocab, 
                                              node_vocab=self.node_vocab, 
                                              relation_vocab=self.relation_vocab)
            self.sub_target_net = KG_Manipulation(config=self.config, 
                                              word_vocab=self.word_vocab, 
                                              node_vocab=self.node_vocab, 
                                              relation_vocab=self.relation_vocab)
            self.meta_target_net.train()
            self.sub_target_net.train()
            self.update_target_net() # update both meta and sub
            for param in self.meta_target_net.parameters():
                param.requires_grad = False
            for param in self.sub_target_net.parameters():
                param.requires_grad = False
            if self.use_cuda:
                self.meta_target_net.cuda()
                self.sub_target_net.cuda()
        else:
            raise Exception("Do not allow non-rl tasks!")

        # 4. optimizer
        # meta
        meta_param_frozen_list = [] # should be changed into torch.nn.ParameterList()
        meta_param_active_list = [] # should be changed into torch.nn.ParameterList()
        for k, v in self.meta_net.named_parameters():
            keep_this = True
            for keyword in self.fix_parameters_keywords:
                if keyword in k:
                    meta_param_frozen_list.append(v)
                    keep_this = False
                    break
            if keep_this:
                meta_param_active_list.append(v)
        meta_param_frozen_list = torch.nn.ParameterList(meta_param_frozen_list)
        meta_param_active_list = torch.nn.ParameterList(meta_param_active_list)
        # sub
        sub_param_frozen_list = [] 
        sub_param_active_list = []
        for k, v in self.sub_net.named_parameters():
            keep_this = True
            for keyword in self.fix_parameters_keywords:
                if keyword in k:
                    sub_param_frozen_list.append(v)
                    keep_this = False
                    break
            if keep_this:
                sub_param_active_list.append(v)
        sub_param_frozen_list = torch.nn.ParameterList(sub_param_frozen_list)
        sub_param_active_list = torch.nn.ParameterList(sub_param_active_list)
        # optimizer
        if self.step_rule == 'adam':
            self.meta_optimizer = torch.optim.Adam(
                                              [{'params': meta_param_frozen_list, 'lr': 0.0},
                                               {'params': meta_param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                               lr=self.config['general']['training']['optimizer']['learning_rate'])
            self.sub_optimizer = torch.optim.Adam(
                                              [{'params': sub_param_frozen_list, 'lr': 0.0},
                                               {'params': sub_param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                               lr=self.config['general']['training']['optimizer']['learning_rate'])
        elif self.step_rule == 'radam':
            from radam import RAdam
            self.meta_optimizer = RAdam(
                                   [{'params': meta_param_frozen_list, 'lr': 0.0},
                                    {'params': meta_param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                    lr=self.config['general']['training']['optimizer']['learning_rate'])
            self.sub_optimizer = RAdam(
                                   [{'params': sub_param_frozen_list, 'lr': 0.0},
                                    {'params': sub_param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                    lr=self.config['general']['training']['optimizer']['learning_rate'])
        else:
            raise NotImplementedError

        # 5. enable accumulative memory for KG
        self.accumulative_counting_memory = AccumulativeCountingMemory()


    def load_config(self):
        self.task = self.config['general']['task']
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # node vocab
        self.node_vocab = []
        with codecs.open("./vocabularies/node_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.node_vocab.append(line.strip().lower())
        self.node2id = {}
        for i, w in enumerate(self.node_vocab):
            self.node2id[w] = i
        # relation vocab
        self.relation_vocab = []
        with codecs.open("./vocabularies/relation_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.relation_vocab.append(line.strip().lower())
        self.origin_relation_number = len(self.relation_vocab)
        # add reverse relations
        for i in range(self.origin_relation_number):
            self.relation_vocab.append(self.relation_vocab[i] + "_reverse")
        # add self relation
        self.relation_vocab += ["self"]
        self.relation2id = {}
        for i, w in enumerate(self.relation_vocab):
            self.relation2id[w] = i
        self.step_rule = self.config['general']['training']['optimizer']['step_rule']
        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.learning_rate_warmup_until = self.config['general']['training']['optimizer']['learning_rate_warmup_until']
        self.fix_parameters_keywords = list(set(self.config['general']['training']['fix_parameters_keywords']))
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']
        self.patience = self.config['general']['training']['patience']
        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_g_belief = self.config['general']['evaluate']['g_belief']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']
        self.max_target_length = self.config['general']['evaluate']['max_target_length']
        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False
        self.experiment_tag = self.config['general']['checkpoint']['experiment_tag']
        self.save_frequency = self.config['general']['checkpoint']['save_frequency']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']
        self.load_graph_update_model_from_tag = self.config['general']['checkpoint']['load_graph_update_model_from_tag']
        self.load_parameter_keywords = list(set(self.config['general']['checkpoint']['load_parameter_keywords']))
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        # RL specific
        self.fully_observable_graph = self.config['rl']['fully_observable_graph']
        # Epsilon greedy: meta and sub only differ in "epsilon_anneal_episodes"
        self.meta_epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['meta_epsilon_anneal_episodes']
        self.sub_epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['sub_epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.meta_epsilon = self.epsilon_anneal_from
        self.sub_epsilon = self.epsilon_anneal_from
        # Disable noisy net
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            raise Exception("Do not allow noisy net")
        # drqn (we don't ust it)
        self.replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']
        self.buffer_reward_threshold = self.config['rl']['replay']['buffer_reward_threshold']
        self.prioritized_replay_beta = self.config['rl']['replay']['prioritized_replay_beta']
        self.beta_scheduler = LinearSchedule(schedule_timesteps=self.max_episode, initial_p=self.prioritized_replay_beta, final_p=1.0)
        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.prioritized_replay_eps = self.config['rl']['replay']['prioritized_replay_eps']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.graph_reward_lambda = self.config['rl']['replay']['graph_reward_lambda']
        self.graph_reward_type = self.config['rl']['replay']['graph_reward_type']
        self.discount_gamma_graph_reward = self.config['rl']['replay']['discount_gamma_graph_reward']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        # replay buffer and updates: the meta and sub memory only differ in size 
        self.meta_dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(
                                            self.config['rl']['replay']['meta_replay_memory_capacity'],
                                            priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                            discount_gamma_game_reward=self.discount_gamma_game_reward,
                                            discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                                            discount_gamma_count_reward=self.discount_gamma_count_reward,
                                            accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.sub_dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(
                                            self.config['rl']['replay']['sub_replay_memory_capacity'],
                                            priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                            discount_gamma_game_reward=self.discount_gamma_game_reward,
                                            discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                                            discount_gamma_count_reward=self.discount_gamma_count_reward,
                                            accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        # input in rl training
        self.enable_recurrent_memory = self.config['rl']['model']['enable_recurrent_memory']
        self.enable_graph_input = self.config['rl']['model']['enable_graph_input']
        self.enable_text_input = self.config['rl']['model']['enable_text_input']
        # rl train and eval
        self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']
        self.use_negative_reward = self.config['rl']['training']['use_negative_reward']
        self.eval_max_nb_steps_per_episode = self.config['rl']['evaluate']['max_nb_steps_per_episode']
        # I assume there's no parallel running environment
        assert (self.batch_size == 1 and self.eval_batch_size == 1), "Do not allow parallel envs!"
        assert self.enable_graph_input
        assert not self.enable_text_input
        assert not self.enable_recurrent_memory

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.meta_net.train()
        self.sub_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.meta_net.eval()
        self.sub_net.eval()

    def update_target_net(self):
        if self.meta_target_net is not None:
            self.meta_target_net.load_state_dict(self.meta_net.state_dict())
        if self.sub_target_net is not None:
            self.sub_target_net.load_state_dict(self.sub_net.state_dict())
    
    def reset_noise(self):
        raise NotImplementedError

    def zero_noise(self):
        raise NotImplementedError

    def load_pretrained_command_generation_model(self, load_from):
        raise NotImplementedError
 
    def load_pretrained_model(self, load_from, load_partial_graph=True):
        meta_load_from = "{}_meta.pt".format(load_from.split(".pt")[0])
        print("Loading meta from: {}".format(meta_load_from))
        print("Loading sub  from: {}\n".format(load_from))
        try:
            if self.use_cuda:
                meta_pretrained_dict = torch.load(meta_load_from)
                sub_pretrained_dict = torch.load(load_from)
            else:
                meta_pretrained_dict = torch.load(meta_load_from, map_location='cpu')
                sub_pretrained_dict = torch.load(load_from, map_location='cpu')
            meta_model_dict = self.meta_net.state_dict()
            sub_model_dict = self.sub_net.state_dict()
            meta_pretrained_dict = {k: v for k, v in meta_pretrained_dict.items() if k in meta_model_dict}
            sub_pretrained_dict = {k: v for k, v in sub_pretrained_dict.items() if k in sub_model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                # meta
                meta_tmp_pretrained_dict = {}
                for k, v in meta_pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            meta_tmp_pretrained_dict[k] = v
                            break
                meta_pretrained_dict = meta_tmp_pretrained_dict
                # sub
                sub_tmp_pretrained_dict = {}
                for k, v in sub_pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            sub_tmp_pretrained_dict[k] = v
                            break
                sub_pretrained_dict = sub_tmp_pretrained_dict
            # update model
            meta_model_dict.update(meta_pretrained_dict)
            self.meta_net.load_state_dict(meta_model_dict)
            sub_model_dict.update(sub_pretrained_dict)
            self.sub_net.load_state_dict(sub_model_dict)
        except:
            raise Exception("Fail to load checkpoint from: {}".format(load_from))

    def save_model_to_path(self, save_to):
        meta_save_to = "{}_meta.pt".format(save_to.split(".pt")[0])
        print("Save meta checkpoint to: {}".format(meta_save_to))
        torch.save(self.meta_net.state_dict(), meta_save_to)
        print("Save sub checkpoint to : {}".format(save_to))
        torch.save(self.sub_net.state_dict(), save_to)
        print("----- Save finished!")

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.
        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:
        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;
        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:
        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);
        Example:
            Here is an example of how to request information and retrieve it.
            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])
        Notes:
            The following information *won't* be available at test time:
            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def init(self):
        pass

    def get_word_input(self, input_strings):
        word_list = [item.split() for item in input_strings]
        word_id_list = [_words_to_ids(tokens, self.word2id) for tokens in word_list]
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_graph_adjacency_matrix(self, triplets):
        adj = np.zeros((len(triplets), len(self.relation_vocab), len(self.node_vocab), len(self.node_vocab)), dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id, node2_id, relation_id = _word_to_id(node1, self.node2id), _word_to_id(node2, self.node2id), _word_to_id(relation, self.relation2id)
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            # self relation
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        adj = to_pt(adj, self.use_cuda, type='float')
        return adj
    
    def get_graph_node_name_input(self):
        res = copy.copy(self.node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self.relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name

    def get_action_candidate_list_input(self, action_candidate_list):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i])
            input_action_candidate_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_action_candidate_list])
        input_action_candidate = np.zeros((batch_size, max_num_candidate, max_word_num))
        input_action_candidate = to_pt(input_action_candidate, self.use_cuda, type="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].size(0), :input_action_candidate_list[i].size(1)] = input_action_candidate_list[i]
        return input_action_candidate

    def choose_model(self, use_model, model_type):
        assert self.task == "rl"
        assert use_model in {"online", "target"}, "use_model should be either online or target"
        assert model_type in {"sub", "meta"}, "wrong model_type: {}".format(model_type)
        if use_model == "online":
            if model_type == "meta":
                model = self.meta_net
            else:
                model = self.sub_net
        else:
            if model_type == "meta":
                model = self.meta_target_net
            else:
                model = self.sub_target_net
        return model

    def encode_graph(self, triplets, use_model, model_type):
        assert model_type in {"sub", "meta"}
        model = self.choose_model(use_model, model_type)
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        adjacency_matrix = self.get_graph_adjacency_matrix(triplets)
        node_encoding_sequence, node_mask = model.encode_graph(input_node_name, input_relation_name, adjacency_matrix)
        return node_encoding_sequence, node_mask
  
    def encode_task(self, tasks, use_model, model_type):
        """
        tasks is a batch list of strings --> encoding like text
        the output is similar to obs_encoding_sequence with mask
        """
        if model_type == "meta":
            return None, None
        model = self.choose_model(use_model, model_type)
        input_tasks = self.get_word_input(tasks)        
        tasks_encoding_sequence, tasks_mask = model.encode_task(input_tasks)
        return tasks_encoding_sequence, tasks_mask

    def encode(self, observation_strings, triplets, tasks, use_model, model_type):
        assert self.task == "rl"
        assert model_type in {"meta", "sub"}
        # 1. encode text
        assert (not self.enable_text_input)
        obs_encoding_sequence, obs_mask = None, None
        # 2. encode graph
        assert (self.enable_graph_input)
        node_encoding_sequence, node_mask = self.encode_graph(triplets, 
                                                              use_model=use_model, 
                                                              model_type=model_type)
        # 3. encode task, note that the meta model will return None here
        tasks_encoding_sequence, tasks_mask = self.encode_task(tasks, 
                                                               use_model=use_model, 
                                                               model_type=model_type)
        return obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask, tasks_encoding_sequence, tasks_mask

    def action_scoring(self, action_candidate_list, 
                       h_og=None, obs_mask=None, h_go=None, node_mask=None, h_tasks=None, tasks_mask=None, 
                       previous_h=None, previous_c=None, use_model="online", 
                       model_type=None):
        assert model_type in {"meta", "sub"}
        model = self.choose_model(use_model, model_type)
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)
        action_scores, action_masks, new_h, new_c = model.score_actions(input_action_candidate, 
                                                                h_og, obs_mask, 
                                                                h_go, node_mask, 
                                                                h_tasks, tasks_mask, 
                                                                previous_h, previous_c)  # batch x num_actions
        return action_scores, action_masks, new_h, new_c

    def act_greedy(self, observation_strings, triplets, action_candidate_list, tasks, previous_h=None, previous_c=None, 
                   model_type=None):
        assert model_type in {"meta", "sub"}
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                                triplets, 
                                                                                tasks, 
                                                                                use_model="online",
                                                                                model_type=model_type) 
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                                h_og, obs_mask, 
                                                                                h_go, node_mask, 
                                                                                h_tasks, tasks_mask, 
                                                                                previous_h, previous_c, use_model="online",
                                                                                model_type=model_type) 
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            replay_info = [observation_strings, triplets, action_candidate_list, tasks, chosen_indices] 
            return chosen_actions, chosen_indices, replay_info, new_h, new_c

    def act_random(self, observation_strings, triplets, action_candidate_list, tasks, previous_h=None, previous_c=None,
                   model_type=None):
        assert model_type in {"meta", "sub"}
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                                triplets, 
                                                                                tasks, 
                                                                                use_model="online",
                                                                                model_type=model_type) 
            action_scores, _, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                    h_og, obs_mask, 
                                                                    h_go, node_mask, 
                                                                    h_tasks, tasks_mask, 
                                                                    previous_h, previous_c, use_model="online",
                                                                    model_type=model_type) 
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)
            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            replay_info = [observation_strings, triplets, action_candidate_list, tasks, chosen_indices]
            return chosen_actions, chosen_indices, replay_info, new_h, new_c

    def act(self, observation_strings, triplets, action_candidate_list, tasks, previous_h=None, previous_c=None, random=False,
            model_type=None):
        assert model_type in {"meta", "sub"}
        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(observation_strings, triplets, action_candidate_list, tasks, previous_h, previous_c, model_type)
            if random:
                return self.act_random(observation_strings, triplets, action_candidate_list, tasks, previous_h, previous_c, model_type)
            # step 1: encode
            batch_size = len(observation_strings)
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                                triplets, 
                                                                                tasks, 
                                                                                use_model="online",
                                                                                model_type=model_type) 
            # step 2: score actions
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                                h_og, obs_mask, 
                                                                                h_go, node_mask, 
                                                                                h_tasks, tasks_mask, 
                                                                                previous_h, previous_c, use_model="online",
                                                                                model_type=model_type) 
            # step 3: choose action indices
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            if model_type == "meta":
                less_than_epsilon = (rand_num < self.meta_epsilon).astype("float32")  # batch
            else:
                less_than_epsilon = (rand_num < self.sub_epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon
            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            # step 4: build replay_info
            replay_info = [observation_strings, triplets, action_candidate_list, tasks, chosen_indices] 
            return chosen_actions, chosen_indices, replay_info, new_h, new_c

    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    
    def get_dqn_loss(self, episode_no, model_type):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        assert model_type in {"meta", "sub"}
        if model_type == "meta":
            if len(self.meta_dqn_memory) < self.replay_batch_size:
                return None, None
            data = self.meta_dqn_memory.sample(self.replay_batch_size, 
                                                beta=self.beta_scheduler.value(episode_no), 
                                                multi_step=self.multi_step)
            if data is None:
                return None, None
        else:
            if len(self.sub_dqn_memory) < self.replay_batch_size:
                return None, None
            data = self.sub_dqn_memory.sample(self.replay_batch_size, 
                                                beta=self.beta_scheduler.value(episode_no), 
                                                multi_step=self.multi_step)
            if data is None:
                return None, None
       
        obs_list, candidate_list, tasks, action_indices, graph_triplet_list, rewards, next_obs_list, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data
        h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(obs_list, 
                                                                            graph_triplet_list, 
                                                                            tasks, 
                                                                            use_model="online",
                                                                            model_type=model_type) 
        action_scores, _, _, _ = self.action_scoring(candidate_list, 
                                                     h_og, obs_mask, 
                                                     h_go, node_mask, 
                                                     h_tasks, tasks_mask, 
                                                     None, None, use_model="online",
                                                     model_type=model_type) 
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                raise Exception("no noisy net")
            # pns Probabilities p(s_t+n, ·; θonline)
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(next_obs_list, 
                                                                                next_graph_triplet_list, 
                                                                                tasks, 
                                                                                use_model="online",
                                                                                model_type=model_type)
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, 
                                                                              h_og, obs_mask, 
                                                                              h_go, node_mask, 
                                                                              h_tasks, tasks_mask,
                                                                              None, None, use_model="online",
                                                                              model_type=model_type)
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(next_obs_list, 
                                                                                next_graph_triplet_list, 
                                                                                tasks, # Here I still use tasks, no tasks_next!
                                                                                use_model="target",
                                                                                model_type=model_type)
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, 
                                                                            h_og, obs_mask, 
                                                                            h_go, node_mask, 
                                                                            h_tasks, tasks_mask, 
                                                                            None, None, use_model="target",
                                                                            model_type=model_type)
            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  # batch
        prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)
        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        if model_type == "meta":
            self.meta_dqn_memory.update_priorities(actual_indices, new_priorities)
        else:
            self.sub_dqn_memory.update_priorities(actual_indices, new_priorities)
        return loss, q_value

    def update_dqn(self, episode_no, model_type):
        assert model_type in {"meta", "sub"}
        assert not self.enable_recurrent_memory, "no recurrent memory"
        dqn_loss, q_value = self.get_dqn_loss(episode_no, model_type)
        if dqn_loss is None:
            return None, None
        # update based on model_type
        if model_type == "meta":
            self.meta_net.zero_grad()
            self.meta_optimizer.zero_grad()
            dqn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), self.clip_grad_norm)
            self.meta_optimizer.step()
        else:
            self.sub_net.zero_grad()
            self.sub_optimizer.zero_grad()
            dqn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sub_net.parameters(), self.clip_grad_norm)
            self.sub_optimizer.step()
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    def get_ingredient_list(self, initial_triplets):
        self.ingredient_list_batch = [kg_utils.get_ingredients(t) for t in initial_triplets]
        # here I assume there's only one env
        batch_size = len(initial_triplets)                                       
        assert (batch_size == 1), "Assume that there's only one env, no parallel"
        self.available_task_list = [None] * batch_size                 # Should be a list of lists
        self.available_task_type_list = [None] * batch_size            # Should be a list of lists
        self.curr_chosen_tasks =      [None] * batch_size              # Should be a list of strings
        self.curr_chosen_task_types = [None] * batch_size              # Should be a list of tuples
        
    def update_task_candidate_list(self, current_triplets):
        available_task_list = []
        available_task_type_list = []
        batch_size = len(current_triplets)
        for i in range(batch_size):
            available_tasks, available_task_types = kg_utils.get_available_tasks(self.ingredient_list_batch[i],
                                                                                 current_triplets[i])
            available_tasks_preproc = [preproc(item, tokenizer=self.nlp) for item in available_tasks]
            available_task_list.append(available_tasks_preproc)
            available_task_type_list.append(available_task_types)
        self.available_task_list = available_task_list
        self.available_task_type_list = available_task_type_list

    def update_chosen_task_type(self, chosen_tasks, chosen_task_indices):
        batch_size = len(chosen_tasks)
        assert (batch_size == 1), "Assume that there's only one env, no parallel"
        for i in range(batch_size):
            curr_chosen_task = chosen_tasks[i]
            curr_chosen_task_index = chosen_task_indices[i]
            curr_available_task_list = self.available_task_list[i]
            curr_available_task_type_list = self.available_task_type_list[i]
            assert (curr_chosen_task == curr_available_task_list[curr_chosen_task_index])
            self.curr_chosen_tasks[i] = curr_chosen_task
            self.curr_chosen_task_types[i] = curr_available_task_type_list[curr_chosen_task_index]

    def get_task_rewards(self, current_triplets):
        task_rewards = []
        whether_new_tasks = []
        batch_size = len(current_triplets)
        for i in range(batch_size):
            assert self.curr_chosen_task_types[i] is not None, "chosen task type should not be None!"
            task_status = kg_utils.check_task_status(self.curr_chosen_task_types[i], current_triplets[i])
            # Get task rewards
            if task_status == 1:
                task_rewards.append(1.)
            else:
                task_rewards.append(0.)
            # Check whether to generate a new task
            if task_status == 0: # Still not solved
                whether_new_tasks.append(False)
            else: # Finished (1) or failed (-1)
                whether_new_tasks.append(True)
        return task_rewards, whether_new_tasks

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network: both meta and sub
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        # update meta episilon 
        if episode_no < self.meta_epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.meta_epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.meta_epsilon_anneal_episodes)
            self.meta_epsilon = max(self.meta_epsilon, 0.0)
        # update meta episilon
        if episode_no < self.sub_epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.sub_epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.sub_epsilon_anneal_episodes)
            self.sub_epsilon = max(self.sub_epsilon, 0.0)

    def get_game_info_at_certain_step_fully_observable(self, obs, infos):
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)
        current_triplets = []
        for b in range(batch_size):
            new_f = set(process_fully_obs_facts(infos["game"][b], infos["facts"][b]))
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
        return observation_strings, current_triplets, action_candidate_list, None, None

    def get_game_info_at_certain_step(self, obs, infos, prev_actions=None, prev_facts=None, return_gt_commands=False):
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)
        new_facts = []
        current_triplets = []  
        commands_from_env = []  
        for b in range(batch_size):
            if prev_facts is None:
                new_f = process_facts(None, infos["game"][b], infos["facts"][b], None, None)
                prev_f = set()
            else:
                new_f = process_facts(prev_facts[b], infos["game"][b], infos["facts"][b], infos["last_action"][b], prev_actions[b])
                prev_f = prev_facts[b]
            new_facts.append(new_f)
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
            target_commands = gen_graph_commands(new_f - prev_f, cmd="add") + gen_graph_commands(prev_f - new_f, cmd="delete")
            commands_from_env.append(target_commands)
        target_command_strings = []
        if return_gt_commands:
            target_command_strings = [" <sep> ".join(sort_target_commands(tgt_cmds)) for tgt_cmds in commands_from_env]
        return observation_strings, current_triplets, action_candidate_list, target_command_strings, new_facts

    def update_knowledge_graph_triplets(self, triplets, prediction_strings):
        new_triplets = []
        for i in range(len(triplets)):
            predict_cmds = prediction_strings[i].split("<sep>")
            if predict_cmds[-1].endswith("<eos>"):
                predict_cmds[-1] = predict_cmds[-1][:-5].strip()
            else:
                predict_cmds = predict_cmds[:-1]
            if len(predict_cmds) == 0:
                new_triplets.append(triplets[i])
                continue
            predict_cmds = [" ".join(item.split()) for item in predict_cmds]
            predict_cmds = [item for item in predict_cmds if len(item) > 0]
            new_triplets.append(update_graph_triplets(triplets[i], predict_cmds, self.node_vocab, self.relation_vocab))
        return new_triplets

    def get_graph_rewards(self, prev_triplets, current_triplets):
        batch_size = len(current_triplets)
        if self.graph_reward_lambda == 0:
            return [0.0 for _ in current_triplets]
        if self.graph_reward_type == "triplets_increased":
            raise Exception("Does not allow triplets_increased")
            rewards = [float(len(c_triplet) - len(p_triplet)) for p_triplet, c_triplet in zip(prev_triplets, current_triplets)]
        elif self.graph_reward_type == "triplets_difference":
            raise Exception("Does not allow triplets_difference")
            rewards = []
            for b in range(batch_size):
                curr = current_triplets[b]
                prev = prev_triplets[b]
                curr = set(["|".join(item) for item in curr])
                prev = set(["|".join(item) for item in prev])
                diff_num = len(prev - curr) + len(curr - prev)
                rewards.append(float(diff_num))
        elif self.graph_reward_type == "KGcount_acc":
            counts = self.accumulative_counting_memory.get_counts(current_triplets)
            rewards = [1. / item for item in counts]
        elif self.graph_reward_type == "KGcount_epi":
            rewards = [1. for _ in current_triplets]
        elif self.graph_reward_type == "KGcount_bebold":
            current_counts = self.accumulative_counting_memory.get_counts(current_triplets)
            prev_counts = self.accumulative_counting_memory.get_counts(prev_triplets)
            rewards = [(1. / current_count - 1. / prev_count) for (prev_count, current_count) in zip(prev_counts, current_counts)]
        else:
            raise NotImplementedError
        rewards = [min(1.0, max(0.0, float(item) * self.graph_reward_lambda)) for item in rewards]
        return rewards

    def reset_binarized_counter(self, batch_size):
        raise NotImplementedError

    def get_binarized_count(self, observation_strings, update=True):
        raise NotImplementedError



