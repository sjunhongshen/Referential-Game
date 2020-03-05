import sys
import os

import numpy as np 
import tensorflow as tf

from pprint import pprint as brint
import pdb

from train import interact, restore_ckpt

from Agents.teacher import Teacher
from Agents.student import Student

from tqdm import tqdm

def main():
    exp_folder = sys.argv[1]
    if len(sys.argv) == 2:
        config_type = ''
    elif sys.argv[2] == 'pretrain':
        config_type = '_pretrain'
    else:
        print('Invalid training phase')
        exit()

    if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
        print('Cannot find target folder')
        exit()
    if (not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')) and config_type == '')\
        or (not os.path.exists(os.path.join('./Experiments', exp_folder, 'config_pretrain.py'))
            and config_type == '_pretrain'):
        print('Cannot find config.py in target folder')
        exit()

    exec('from Experiments.%s.config%s import train_config, game_config, vision_config, belief_config, value_config'
         % (exp_folder, config_type), globals())
    train_config.save_dir = os.path.join('./Experiments', exp_folder)
    game_config.save_dir = train_config.save_dir

    if train_config.device.find('cpu') == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)

    with tf.device(train_config.device):
        teacher = Teacher(sess, train_config, game_config, vision_config, belief_config, value_config)
        student = Student(sess, train_config, game_config, vision_config, belief_config, value_config)

    init = tf.global_variables_initializer()
    sess.run(init)

    np.random.seed(425)
    exec('from %s.concept import Concept' % game_config.dataset, globals())
    
    concept_generator = Concept(game_config)
    # teacher.pretrain_bayesian_belief_update(concept_generator)
    # student.pretrain_bayesian_belief_update(concept_generator)

    ckpt_path = os.path.join(train_config.save_dir, 'TOM_CL_phase3_5000')
    restored = restore_ckpt(student, teacher, ckpt_path)
    if not restored:
        print('Load from %s failed' % ckpt_path)
        exit(1)
    accuracies = []
    #accuracies_rtd_1 = []
    rewards = []
    test_rounds = 100
    hard_correct = [0, 0]
    rtd_1_correct = [0, 0]
    rtd_1_hard_correct = [0, 0]
    all_wrongs = []
    td_1_wrong = 0
    td_2_wrong = 0
    td_3_wrong = 0
    td_4_wrong = 0
    len_nb = 0
    len_nb_wrong = 0
    len_wrong = 0
    len_invalid_message = 0
    len_invalid_message_wrong = 0
    len_msg_not_unique_id = 0
    len_msg_not_unique_id_wrong = 0
    len_msg_not_unique_id_level_2 = 0
    len_msg_not_unique_id_wrong_level_2 = 0
    len_msg_unique_id_level_2 = 0
    len_msg_unique_id_wrong_level_2 = 0
    len_msg_not_unique_id_level_3 = 0
    len_msg_not_unique_id_wrong_level_3 = 0
    len_msg_unique_id_level_3 = 0
    len_msg_unique_id_wrong_level_3 = 0
    len_td_1 = 0
    len_td_2 = 0
    len_td_3 = 0
    len_td_4 = 0
    len_td_2_rtd_1 = 0
    len_rtd_1 = 0
    len_td_2_rtd_1_wrong = 0
    len_subset = 0
    len_subset_wrong = 0
    len_subset_nb = 0
    len_subset_nb_wrong = 0

    rtd_1_wrong = 0
    level_1 = 0
    level_2 = 0
    level_3 = 0
    level_4 = 0
    level_1_wrong = 0
    level_2_wrong = 0
    level_3_wrong = 0
    level_4_wrong = 0
    rtd_more_than_1 = 0
    rtd_more_than_1_wrong = 0
    student_wrongs = []
    right_q_values_level_1_p1 = []
    right_q_values_level_2_p1 = []
    wrong_q_values_level_1_p1 = []
    wrong_q_values_level_2_p1 = []
    q_values_msg_level_2_unique_id = []
    q_values_msg_level_2_not_unique_id = []
    pos_to_msg = dict([(i, np.zeros(game_config.message_space_size)) for i in range(game_config.num_distractors)])


    epsilon = 0
    for tr in tqdm(range(test_rounds)):
        trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
            interact(teacher, student, concept_generator, epsilon, 0.9, None, 100, student_think = True)
        accuracies.append(accuracy)
        #accuracies_rtd_1.append(accuracy_rtd_1)
        rewards.append(reward)
        long_batch = [db for db in debug_batch if len(db) > 1]
        g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
        b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
        bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
        bad_batch_td_1 = [lb for lb in bad_batch if lb[0]['teaching_dim'] == 1]
        bad_batch_td_2 = [lb for lb in bad_batch if lb[0]['teaching_dim'] == 2]
        bad_batch_td_3 = [lb for lb in bad_batch if lb[0]['teaching_dim'] == 3]
        bad_batch_td_4 = [lb for lb in bad_batch if lb[0]['teaching_dim'] == 4]
        bad_batch_nb = [lb for lb in bad_batch if lb[0]['non_bayesian'] > 0]
        nb_batch = [db for db in debug_batch if db[0]['non_bayesian'] > 0]
        hard_batch = [db for db in debug_batch if db[0]['teaching_dim'] > 1]
        g_hard_batch = [lb for lb in hard_batch if lb[0]['gain'] > 0]
        b_hard_batch = [lb for lb in hard_batch if lb[0]['gain'] < 0]
        for db in debug_batch:
            pos_to_msg[db[0]['tea_target_idx']][db[0]['message']] += 1

        subset_batch = [db for db in debug_batch if sum([set(np.array(db[0]['concepts'][i])).issubset(set(np.array(np.array(db[0]['concepts'][j])))) for i in range(game_config.num_distractors) for j in range(game_config.num_distractors) if i != j]) > 0]
        len_subset += len(subset_batch)

        subset_wrong_batch = [lb for lb in subset_batch if lb[0]['gain'] < 0]
        len_subset_wrong += len(subset_wrong_batch)

        subset_nb_batch = [lb for lb in subset_batch if lb[0]['non_bayesian'] > 0]
        len_subset_nb += len(subset_nb_batch)

        subset_nb_wrong_batch = [lb for lb in subset_nb_batch if lb[0]['gain'] < 0]
        len_subset_nb_wrong += len(subset_nb_wrong_batch)

        other_nb_batch = [db for db in nb_batch if sum([set(np.array(db[0]['concepts'][i])).issubset(set(np.array(np.array(db[0]['concepts'][j])))) for i in range(game_config.num_distractors) for j in range(game_config.num_distractors) if i != j]) == 0 ]


        rtd_1_batch = [db for db in debug_batch if db[0]['recursive_teaching_dim'] == 1]
        g_rtd_1_batch = [lb for lb in rtd_1_batch if lb[0]['gain'] > 0]
        b_rtd_1_batch = [lb for lb in rtd_1_batch if lb[0]['gain'] < 0]
        rtd_1_correct[0] += len(g_rtd_1_batch)
        rtd_1_correct[1] += len(rtd_1_batch)
        rtd_1_wrong += len(b_rtd_1_batch)

        #new check by level
        rtd_more_than_1_batch = [db for db in debug_batch if db[0]['recursive_teaching_dim'] > 1]
        b_rtd_more_than_1_batch = [db for db in rtd_more_than_1_batch if db[0]['gain'] < 0]
        rtd_1_level_1_batch = [db for db in rtd_1_batch if db[0]['level_of_target'][0] == 1]
        rtd_1_level_2_batch = [db for db in rtd_1_batch if db[0]['level_of_target'][0] == 2]
        rtd_1_level_3_batch = [db for db in rtd_1_batch if db[0]['level_of_target'][0] == 3]
        rtd_1_level_4_batch = [db for db in rtd_1_batch if db[0]['level_of_target'][0] == 4]

        g_rtd_1_level_1_batch = [db for db in g_rtd_1_batch if db[0]['level_of_target'][0] == 1]
        g_rtd_1_level_2_batch = [db for db in g_rtd_1_batch if db[0]['level_of_target'][0] == 2]
        g_rtd_1_level_3_batch = [db for db in g_rtd_1_batch if db[0]['level_of_target'][0] == 3]
        g_rtd_1_level_4_batch = [db for db in g_rtd_1_batch if db[0]['level_of_target'][0] == 4]

        b_rtd_1_level_1_batch = [db for db in b_rtd_1_batch if db[0]['level_of_target'][0] == 1]
        b_rtd_1_level_2_batch = [db for db in b_rtd_1_batch if db[0]['level_of_target'][0] == 2]
        b_rtd_1_level_3_batch = [db for db in b_rtd_1_batch if db[0]['level_of_target'][0] == 3]
        b_rtd_1_level_4_batch = [db for db in b_rtd_1_batch if db[0]['level_of_target'][0] == 4]

        level_1 += len(rtd_1_level_1_batch)
        level_2 += len(rtd_1_level_2_batch)
        level_3 += len(rtd_1_level_3_batch)
        level_4 += len(rtd_1_level_4_batch)
        level_1_wrong += len(b_rtd_1_level_1_batch)
        level_2_wrong += len(b_rtd_1_level_2_batch)
        level_3_wrong += len(b_rtd_1_level_3_batch)
        level_4_wrong += len(b_rtd_1_level_4_batch)
        rtd_more_than_1 += len(rtd_more_than_1_batch)
        rtd_more_than_1_wrong += len(b_rtd_more_than_1_batch)

        if debug_batch[0][0]['stu_msg_est'] is not None:
            for idx, db in enumerate(debug_batch):
                target_id = int(np.nonzero(db[0]['teacher'])[0])
                if db[0]['stu_msg_est'][0, target_id, db[0]['message']] < 0.1:
                    student_wrongs.append(db)
        else:
            student_wrongs = None

        rtd_1_hard_batch = [db for db in rtd_1_batch if db[0]['teaching_dim'] > 1]
        g_rtd_1_hard_batch = [lb for lb in rtd_1_hard_batch if lb[0]['gain'] > 0]
        rtd_1_hard_correct[0] += len(g_rtd_1_hard_batch)
        rtd_1_hard_correct[1] += len(rtd_1_hard_batch)

        hard_correct[0] += len(g_hard_batch)
        hard_correct[1] += len(hard_batch)
        all_wrongs.extend(bad_batch)
        
        len_nb += len(nb_batch)
        len_nb_wrong += len(bad_batch_nb)
        len_wrong += len(bad_batch)
        td_1_wrong += len(bad_batch_td_1)
        td_2_wrong += len(bad_batch_td_2)
        td_3_wrong += len(bad_batch_td_3)
        td_4_wrong += len(bad_batch_td_4)

        invalid_message_batch = [db for db in debug_batch if db[0]['message'] not in set(db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])])]
        invalid_message_wrong_batch = [lb for lb in invalid_message_batch if lb[0]['gain']<0 ]
        len_invalid_message += len(invalid_message_batch)
        len_invalid_message_wrong += len(invalid_message_wrong_batch)
        
        td_1_batch = [db for db in debug_batch if db[0]['teaching_dim']==1]
        td_2_batch = [db for db in debug_batch if db[0]['teaching_dim']==2]
        td_3_batch = [db for db in debug_batch if db[0]['teaching_dim']==3]
        td_4_batch = [db for db in debug_batch if db[0]['teaching_dim']==4]
        td_2_rtd_1_batch = [lb for lb in td_2_batch if lb[0]['recursive_teaching_dim'] == 1]
        td_2_rtd_1_wrong_batch = [lb for lb in td_2_rtd_1_batch if lb[0]['gain'] < 0]
        len_td_2_rtd_1 += len(td_2_rtd_1_batch)
        len_td_2_rtd_1_wrong += len(td_2_rtd_1_wrong_batch)
        len_rtd_1 += len(rtd_1_batch)
        len_td_1 += len(td_1_batch)
        len_td_2 += len(td_2_batch)
        len_td_3 += len(td_3_batch)
        len_td_4 += len(td_4_batch)

        #msg_not_unique_id_batch = [db for db in td_1_batch if db[0]['message'] in set([x for y in db[0]['concepts'] for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]]) == 1 ]
        #msg_not_unique_id_wrong_batch = [lb for lb in msg_not_unique_id_batch if lb[0]['gain'] < 0 ]
        '''
        for i in range(11):
            db = rtd_1_level_2_batch[i]
            print(i)
            print(db[0]['concepts'])
            print(db[0]['levels'])
            print(db[0]['message'])
            print(np.multiply(np.array(db[0]['concepts']), np.array([i>1 for i in db[0]['levels']])))
            print(db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])])
            print(set(db[0]['message']).issubset(set([x for y in np.multiply(np.array(db[0]['concepts']), np.array([i>1 for i in db[0]['levels']])) for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]])) == 1)

        pdb.set_trace()
        '''
        
        msg_not_unique_id_level_2_batch = []
        for db in rtd_1_level_2_batch:
            level2_dist_attr = set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 1) * (1 - db[0]['teacher']))[0]], ()))
            if db[0]['message'] in level2_dist_attr:
                msg_not_unique_id_level_2_batch.append(db)
        #msg_not_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if db[0]['message'][0] in set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 1) * (1 - db[0]['teacher']))[0]], ()))]
        #msg_not_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if set(db[0]['message']).issubset(set([x for y in [db[0]['concepts'][i] * (db[0]['levels'][i]>1) for i in range(game_config.num_distractors)] for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]]))]
        msg_not_unique_id_wrong_level_2_batch = [lb for lb in msg_not_unique_id_level_2_batch if lb[0]['gain'] < 0 ]

        msg_unique_id_level_2_batch = []
        for db in rtd_1_level_2_batch:
            level2_dist_attr = set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 1) * (1 - db[0]['teacher']))[0]], ()))
            if db[0]['message'] not in level2_dist_attr:
                msg_unique_id_level_2_batch.append(db)
        #msg_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if set(db[0]['message']).issubset(set([x for y in [db[0]['concepts'][i] * (db[0]['levels'][i]>1) for i in range(game_config.num_distractors)] for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]]))]
        msg_unique_id_wrong_level_2_batch = [lb for lb in msg_unique_id_level_2_batch if lb[0]['gain'] < 0 ]

        msg_not_unique_id_level_3_batch = []
        for db in rtd_1_level_3_batch:
            level3_dist_attr = set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 2) * (1 - db[0]['teacher']))[0]], ()))
            if db[0]['message'] in level3_dist_attr:
                msg_not_unique_id_level_3_batch.append(db)
        #msg_not_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if db[0]['message'][0] in set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 1) * (1 - db[0]['teacher']))[0]], ()))]
        #msg_not_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if set(db[0]['message']).issubset(set([x for y in [db[0]['concepts'][i] * (db[0]['levels'][i]>1) for i in range(game_config.num_distractors)] for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]]))]
        msg_not_unique_id_wrong_level_3_batch = [lb for lb in msg_not_unique_id_level_3_batch if lb[0]['gain'] < 0 ]

        msg_unique_id_level_3_batch = []
        for db in rtd_1_level_3_batch:
            level3_dist_attr = set(sum([db[0]['concepts'][i] for i in np.nonzero((db[0]['levels'] > 2) * (1 - db[0]['teacher']))[0]], ()))
            if db[0]['message'] not in level3_dist_attr:
                msg_unique_id_level_3_batch.append(db)
        #msg_unique_id_level_2_batch = [db for db in rtd_1_level_2_batch if set(db[0]['message']).issubset(set([x for y in [db[0]['concepts'][i] * (db[0]['levels'][i]>1) for i in range(game_config.num_distractors)] for x in y if y!= db[0]['concepts'][int(np.nonzero(db[0]['teacher'])[0])]]))]
        msg_unique_id_wrong_level_3_batch = [lb for lb in msg_unique_id_level_3_batch if lb[0]['gain'] < 0 ]

        #len_msg_not_unique_id += len(msg_not_unique_id_batch)
        #len_msg_not_unique_id_wrong += len(msg_not_unique_id_wrong_batch)
        len_msg_not_unique_id_level_2 += len(msg_not_unique_id_level_2_batch)
        len_msg_not_unique_id_wrong_level_2 += len(msg_not_unique_id_wrong_level_2_batch)
        len_msg_unique_id_level_2 += len(msg_unique_id_level_2_batch)
        len_msg_unique_id_wrong_level_2 += len(msg_unique_id_wrong_level_2_batch)
        len_msg_not_unique_id_level_3 += len(msg_not_unique_id_level_3_batch)
        len_msg_not_unique_id_wrong_level_3 += len(msg_not_unique_id_wrong_level_3_batch)
        len_msg_unique_id_level_3 += len(msg_unique_id_level_3_batch)
        len_msg_unique_id_wrong_level_3 += len(msg_unique_id_wrong_level_3_batch)



        right_q_values_level_1_p1_batch = [np.max([[g_rtd_1_level_1_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(g_rtd_1_level_1_batch))]
        right_q_values_level_1_p1 += right_q_values_level_1_p1_batch
        
        right_q_values_level_2_p1_batch = [np.max([[g_rtd_1_level_2_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(g_rtd_1_level_2_batch))]
        right_q_values_level_2_p1 += right_q_values_level_2_p1_batch
        wrong_q_values_level_1_p1_batch = [np.max([[b_rtd_1_level_1_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(b_rtd_1_level_1_batch))]
        wrong_q_values_level_1_p1 += wrong_q_values_level_1_p1_batch
        wrong_q_values_level_2_p1_batch = [np.max([[b_rtd_1_level_2_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(b_rtd_1_level_2_batch))]            
        wrong_q_values_level_2_p1 += wrong_q_values_level_2_p1_batch

        q_values_msg_level_2_unique_id_batch = [np.max([[msg_unique_id_level_2_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(msg_unique_id_level_2_batch))]
        q_values_msg_level_2_unique_id += q_values_msg_level_2_unique_id_batch
        q_values_msg_level_2_not_unique_id_batch = [np.max([[msg_not_unique_id_level_2_batch[j][0]['msg_Q']][0][i][1] for i in range(game_config.message_space_size)]) for j in range(len(msg_not_unique_id_level_2_batch))]
        q_values_msg_level_2_not_unique_id += q_values_msg_level_2_not_unique_id_batch
        




        #my_specific_example = [db for db in debug_batch if db[0]['concepts'] == [(7,), (0,1,6,7),(0,3,4,5),(4,5,6,7)] ]

        #if len(other_nb_batch) > 0:
        #if len(b_rtd_1_level_2_batch) > 0:
        #len(long_batch) > 0 or len(bad_batch) > 0:
        if len(msg_unique_id_wrong_level_2_batch) > 0:
            pass#pdb.set_trace()
        #pdb.set_trace()

    print(np.mean(accuracies), np.mean(rewards))
    print('hard accuracy:', 1.0 * hard_correct[0] / hard_correct[1])
    print('rtd = 1 accuracy', 1.0 * rtd_1_correct[0] / rtd_1_correct[1])
    print('rtd = 1 hard accuracy', 1.0 * rtd_1_hard_correct[0] / rtd_1_hard_correct[1])
    print('td_1_wrong ' + str(td_1_wrong))
    print('td_2_wrong ' + str(td_2_wrong))
    print('td_3_wrong ' + str(td_3_wrong))
    print('td_4_wrong ' + str(td_4_wrong))
    print('len_nb ' + str(len_nb))
    print('len_nb_wrong ' + str(len_nb_wrong))
    print('len_wrong ' + str(len_wrong))
    print('len_invalid_message ' + str(len_invalid_message) )
    print('len_invalid_message_wrong ' + str(len_invalid_message_wrong))
    #print('len_msg_not_unique_id ' + str(len_msg_not_unique_id))
    #print('len_msg_not_unique_id_wrong ' + str(len_msg_not_unique_id_wrong))
    print('len_td_1 ' + str(len_td_1))
    print('len_td_2 ' + str(len_td_2))
    print('len_td_3 ' + str(len_td_3))
    print('len_td_4 ' + str(len_td_4))
    print('len_td_2_rtd_1 ' + str(len_td_2_rtd_1))
    print('len_rtd_1 ' + str(len_rtd_1))
    print('len_td_2_rtd_1_wrong ' + str(len_td_2_rtd_1_wrong))
    print('len_subset ' + str(len_subset))
    print('len_subset_wrong ' + str(len_subset_wrong))
    print('len_subset_nb ' + str(len_subset_nb))
    print('len_subset_nb_wrong ' + str(len_subset_nb_wrong))
    print('length of rtd 1 examples: ' + str(rtd_1_correct[1]))
    print('length of wrong rtd 1 examples: ' + str(rtd_1_wrong))
    print('length of rtd > 1 examples: ' + str(rtd_more_than_1))
    print('length of wrong rtd > 1 examples: ' + str(rtd_more_than_1_wrong))
    print('level_1 ' + str(level_1))
    print('level_1_wrong ' + str(level_1_wrong))
    print('level_1_accu: %f ' % (1 - 1.0 * level_1_wrong / level_1))
    print('level_2 ' + str(level_2))
    print('level_2_wrong ' + str(level_2_wrong))
    print('level_2_accu: %f ' % (1 - 1.0 * level_2_wrong / level_2))
    print('level_3 ' + str(level_3))
    print('level_3_wrong ' + str(level_3_wrong))
    print('level_3_accu: %f ' % (1 - 1.0 * level_3_wrong / level_3))
    print('level_4 ' + str(level_4))
    print('level_4_wrong ' + str(level_4_wrong))
    print('len_msg_not_unique_id_level_2 ' + str(len_msg_not_unique_id_level_2))
    print('len_msg_not_unique_id_wrong_level_2 ' + str(len_msg_not_unique_id_wrong_level_2))
    print('len_msg_not_unique_id_level_3 ' + str(len_msg_not_unique_id_level_3))
    print('len_msg_not_unique_id_wrong_level_3 ' + str(len_msg_not_unique_id_wrong_level_3))
    if student_wrongs is not None:
        print('student BU wrong %d' % len(student_wrongs))
    print(np.percentile(right_q_values_level_1_p1, [0, 5, 10, 15, 20 ,50,90,100]))
    print(np.percentile(right_q_values_level_2_p1, [0, 5, 10, 15, 20 ,50,90,100]))
    #print(np.percentile(wrong_q_values_level_1_p1, [0, 5, 10, 15, 20 ,50,90,100]))
    print(np.percentile(wrong_q_values_level_2_p1, [0, 5, 10, 15, 20 ,50,90,100]))
    print(np.percentile(q_values_msg_level_2_unique_id, [0, 5, 10, 15, 20 ,50,70, 90,100]))
    print(np.percentile(q_values_msg_level_2_not_unique_id, [0, 5, 10, 15, 20 ,50,70, 90,100]))

    for k in pos_to_msg:
        pos_to_msg[k] = pos_to_msg[k] / np.sum(pos_to_msg[k])
    brint(pos_to_msg)
    #[-0.01712466  0.01551782  0.74689277  0.79215789] 

if __name__ == '__main__':
    main()
    #pdb.set_trace()
