
import numpy as np


def get_ingredients(triplets_full_0):
    result = []
    for triplet in triplets_full_0:
        if triplet[1] == 'cookbook':
            assert(triplet[2] == 'part_of'), "Strange triplet: {}".format(triplet)
            result.append(triplet[0])
    return result

def _check_ing_existance(ingredient, triplets_full):
    for triplet in triplets_full:
        if (triplet[0] == ingredient) and (triplet[2] in {'in', 'on', 'at'}):
            return True
    return False

def _check_ing_collection(ingredient, triplets_full):
    if [ingredient, "player", "in"] in triplets_full:
        return True
    else:
        return False

def _get_ing_req_status(ingredient, triplets_full):
    req_result_part1_temp = []
    req_result_part2_temp = []
    status_result = set()
    for triplet in triplets_full:
        if (triplet[0] == ingredient):
            if (triplet[2] == 'needs'):
                if triplet[1] in {'chopped', 'diced', 'sliced'}:
                    req_result_part1_temp.append(triplet[1])
                else:
                    req_result_part2_temp.append(triplet[1])
            if (triplet[2] == 'is'):
                status_result.add(triplet[1])
    req_result_part1 = []
    req_result_part2 = []
    for req in req_result_part1_temp:
        if req not in status_result:
            req_result_part1.append(req)
    for req in req_result_part2_temp:
        if req not in status_result:
            req_result_part2.append(req)
    return req_result_part1, req_result_part2, status_result



def get_available_tasks(ingredients, triplets_full, test_mode=False):
    req_act_dict = {'chopped':'chop', 
                    'diced':'dice', 
                    'fried':'cook',
                    'grilled':'cook',
                    'roasted':'cook',
                    'sliced':'slice'}
    req_obj_dict = {'chopped':'knife', 
                    'diced':'knife', 
                    'fried':'stove',
                    'grilled':'bbq',
                    'roasted':'oven',
                    'sliced':'knife'}
    available_tasks = []
    available_task_types = []
    for ingredient in ingredients:
        # this ingredient should be in the KG
        if _check_ing_existance(ingredient, triplets_full):
            # 0. get requirement list and status set, here requirement_result contains unfinished requirements only
            req_result_part1, req_result_part2, status_result = _get_ing_req_status(ingredient, triplets_full)
            if test_mode:
                print("Ing {}|Status: {}|Req1: {}|Req2: {}".format(ingredient, status_result, req_result_part1, req_result_part2))
            # 1. not collected --> find X
            if not _check_ing_collection(ingredient, triplets_full):
                generated_task = "find {}".format(ingredient)
                available_tasks.append(generated_task)
                available_task_types.append(("find", ingredient))
            # 2. already collected --> check unfinished requirement
            else:
                if (len(req_result_part1) + len(req_result_part2)) > 0:
                    if len(req_result_part1) > 0:
                        curr_req = req_result_part1[0]
                        if not _check_ing_collection("knife",triplets_full):
                            generated_task = "find knife"
                            available_tasks.append(generated_task)
                            available_task_types.append(("find", "knife"))
                        else:
                            act = req_act_dict[curr_req]
                            obj = req_obj_dict[curr_req]
                            generated_task = "{} {} with {}".format(act, ingredient, obj)
                            available_tasks.append(generated_task)
                            available_task_types.append(("make", ingredient, curr_req))
                    else:
                        for curr_req in req_result_part2:
                            act = req_act_dict[curr_req]
                            obj = req_obj_dict[curr_req]
                            generated_task = "{} {} with {}".format(act, ingredient, obj)
                            available_tasks.append(generated_task)
                            available_task_types.append(("make", ingredient, curr_req))
    # final
    if len(available_tasks) == 0:
        available_tasks.append("prepare meal eat meal")
        available_task_types.append(("prepare", "meal"))
    return available_tasks, available_task_types


def check_task_status(task_type, triplets_full):
    if task_type[0] == "find":
        if _check_ing_collection(task_type[1], triplets_full):
            return 1
        elif _check_ing_existance(task_type[1], triplets_full):
            return 0
        else:
            return -1
    elif task_type[0] == "make":
        if [task_type[1], task_type[2], "is"] in triplets_full:
            return 1
        elif _check_ing_existance(task_type[1], triplets_full):
            return 0
        else:
            return -1
    else:
        assert (task_type[0] == "prepare"), "Only holds for prepare, got {}".format(task_type[0])
        if ['meal', "consumed", "is"] in triplets_full:
            return 1
        else:
            return 0




if __name__ == "__main__":
    # full_triplets_sample = [['player', 'kitchen', 'at'], ['bed', 'bedroom', 'at'], ['counter', 'kitchen', 'at'], ['fridge', 'kitchen', 'at'], ['oven', 'kitchen', 'at'], ['shelf', 'pantry', 'at'], ['sofa', 'livingroom', 'at'], ['stove', 'kitchen', 'at'], ['table', 'kitchen', 'at'], ['toilet', 'bathroom', 'at'], ['yellow potato', 'chopped', 'is'], ['fridge', 'closed', 'is'], ['frosted-glass door', 'closed', 'is'], ['bedroom', 'livingroom', 'east_of'], ['kitchen', 'corridor', 'east_of'], ['livingroom', 'kitchen', 'east_of'], ['block of cheese', 'fridge', 'in'], ['carrot', 'fridge', 'in'], ['water', 'fridge', 'in'], ['yellow potato', 'player', 'in'], ['yellow potato', 'diced', 'needs'], ['yellow potato', 'roasted', 'needs'], ['bathroom', 'corridor', 'north_of'], ['frosted-glass door', 'kitchen', 'north_of'], ['cookbook', 'counter', 'on'], ['knife', 'counter', 'on'], ['purple potato', 'counter', 'on'], ['salt', 'shelf', 'on'], ['yellow bell pepper', 'counter', 'on'], ['salt', 'cookbook', 'part_of'], ['water', 'cookbook', 'part_of'], ['yellow potato', 'cookbook', 'part_of'], ['block of cheese', 'raw', 'is'], ['carrot', 'raw', 'is'], ['yellow bell pepper', 'raw', 'is'], ['yellow potato', 'roasted', 'is'], ['corridor', 'bathroom', 'south_of'], ['frosted-glass door', 'pantry', 'south_of'], ['block of cheese', 'uncut', 'is'], ['carrot', 'uncut', 'is'], ['purple potato', 'uncut', 'is'], ['yellow bell pepper', 'uncut', 'is'], ['corridor', 'kitchen', 'west_of'], ['kitchen', 'livingroom', 'west_of'], ['livingroom', 'bedroom', 'west_of']]
    full_triplets_sample = [['backyard', 'garden', 'east_of'], ['bathroom', 'corridor', 'west_of'], ['bbq', 'backyard', 'at'], ['bed', 'bedroom', 'at'], ['bedroom', 'corridor', 'east_of'], ['bedroom', 'livingroom', 'south_of'], ['block of cheese', 'raw', 'is'], ['block of cheese', 'showcase', 'on'], ['block of cheese', 'uncut', 'is'], ['carrot', 'cookbook', 'part_of'], ['carrot', 'player', 'in'], ['carrot', 'raw', 'is'], ['carrot', 'roasted', 'needs'], ['carrot', 'sliced', 'needs'], ['carrot', 'uncut', 'is'], ['cookbook', 'counter', 'on'], ['corridor', 'bathroom', 'east_of'], ['corridor', 'bedroom', 'west_of'], ['corridor', 'kitchen', 'south_of'], ['counter', 'kitchen', 'at'], ['driveway', 'street', 'north_of'], ['fiberglass door', 'driveway', 'west_of'], ['fiberglass door', 'livingroom', 'east_of'], ['fiberglass door', 'open', 'is'], ['fridge', 'closed', 'is'], ['fridge', 'kitchen', 'at'], ['frosted-glass door', 'closed', 'is'], ['frosted-glass door', 'kitchen', 'north_of'], ['frosted-glass door', 'pantry', 'south_of'], ['garden', 'backyard', 'west_of'], ['kitchen', 'corridor', 'north_of'], ['kitchen', 'livingroom', 'west_of'], ['knife', 'player', 'in'], ['livingroom', 'bedroom', 'north_of'], ['livingroom', 'kitchen', 'east_of'], ['orange bell pepper', 'counter', 'on'], ['orange bell pepper', 'raw', 'is'], ['orange bell pepper', 'uncut', 'is'], ['oven', 'kitchen', 'at'], ['patio chair', 'backyard', 'at'], ['patio table', 'backyard', 'at'], ['player', 'kitchen', 'at'], ['pork chop', 'cookbook', 'part_of'], ['pork chop', 'diced', 'needs'], ['pork chop', 'roasted', 'needs'], ['pork chop', 'showcase', 'on'], ['pork chop', 'uncut', 'is'], ['purple potato', 'counter', 'on'], ['purple potato', 'uncut', 'is'], ['red apple', 'counter', 'on'], ['red apple', 'raw', 'is'], ['red apple', 'uncut', 'is'], ['red hot pepper', 'chopped', 'needs'], ['red hot pepper', 'cookbook', 'part_of'], ['red hot pepper', 'counter', 'on'], ['red hot pepper', 'fried', 'needs'], ['red hot pepper', 'raw', 'is'], ['red hot pepper', 'uncut', 'is'], ['red onion', 'garden', 'at'], ['red onion', 'raw', 'is'], ['red onion', 'uncut', 'is'], ['red potato', 'counter', 'on'], ['red potato', 'uncut', 'is'], ['screen door', 'backyard', 'north_of'], ['screen door', 'corridor', 'south_of'], ['screen door', 'open', 'is'], ['shelf', 'pantry', 'at'], ['showcase', 'supermarket', 'at'], ['sliding door', 'closed', 'is'], ['sliding door', 'street', 'south_of'], ['sliding door', 'supermarket', 'north_of'], ['sofa', 'livingroom', 'at'], ['stove', 'kitchen', 'at'], ['street', 'driveway', 'south_of'], ['table', 'kitchen', 'at'], ['toilet', 'bathroom', 'at'], ['toolbox', 'closed', 'is'], ['toolbox', 'shed', 'at'], ['wooden door', 'backyard', 'south_of'], ['wooden door', 'closed', 'is'], ['wooden door', 'shed', 'north_of'], ['workbench', 'shed', 'at'], ['yellow bell pepper', 'garden', 'at'], ['yellow bell pepper', 'raw', 'is'], ['yellow bell pepper', 'uncut', 'is'], ['yellow potato', 'counter', 'on'], ['yellow potato', 'uncut', 'is']]

    ingredients = get_ingredients(full_triplets_sample)
    print("Ingredients: {}".format(ingredients))
    # print("Task: {}".format(generate_task_multi_ing(ingredients, full_triplets_sample)))
    available_tasks, available_task_types = get_available_tasks(ingredients, full_triplets_sample, test_mode=True)
    for i in range(len(available_tasks)):
        print("{}:{}".format(available_tasks[i], available_task_types[i]))
    print("-----")

    task_type = ('prepare', 'meal')
    print(check_task_status(task_type, full_triplets_sample))
