from hopfieldnetwork import *
from grid import *
from nao_speech import *
from nao_imagecapture import *
from PIL import Image
from audio import *
from recaudio import *
import random
from wn import *

'''This script contains the Sarsa algorithm embedded with robot actions for visual audio game.
 It contains the reward functions that is based on the Hopfield energy and an epsilon greedy state decision policy.
 The visual audio images are concatenated. So the training needs to be done with visual and audio data.'''

def generate_reward(train_images, current_state, current_action, img_c, img_f):
    """ Generates reward based on the energy used in the Hopfield network.
    Parameters
    -----------
    train_images: bipolar array
    the training images for the Hopfield Network
    current_state: integer
    number of chosen image
    current_action: integer
    number of image to move to
    rsize: tuple
    preferred size of the individual pictures

    Return
    -------
    reward : positive reward for low energy images
    energy_state: energy of current state
    energy_action: energy of future state
    """

    # run the image trough the hopfield to generate an energy
    weights_h = calc_weights(train_images)

    filename_1 = constants.store_captured + '%s.png' % img_c

    filename_2 = constants.store_captured + '%s.png' % img_f

    current_pattern = bipolarize_pattern_robot(filename_1)
    future_pattern = bipolarize_pattern_robot(filename_2)
    new_s1, changed_bits1, state_changes1, epochs1 = calc_stateupdate_async(current_pattern, weights_h, 1000, 1000)
    new_s2, changed_bits2, state_changes2, epochs2 = calc_stateupdate_async(future_pattern, weights_h, 1000, 1000)
    energy_state = changed_bits1
    energy_action = changed_bits2
    if energy_state < energy_action:
        reward = -1
    else:
        reward = 1
    #reward = energy_state - energy_action


    return reward, energy_state, energy_action


def update_q(train_images, start_location, start_img, max_iter,  q, cumulative_reward, cumulative_energy, total_reward, total_energy, no_of_state_visits, total_energy_by_state, total_reward_by_state, td_storage, helper_reward, i, condition, help_requests, interaction_stops, interaction_restart):
    """Calculates q value, updates in each iteration based on a specified policy and reward function
     Parameters
    -----------
    prepath: filepath
    the path where the images to display are located
    postpath: filepath
    the path where the captured images are to be stored
    train_images: bipolar array
    the training images for the Hopfield Network
    start_location: integer
    number of starting image on grid
    grid_width: integer
    number of images in the vertical direction of the grid
    grid_length: integer
    number of images in the horizontal direction of the grid
    max_iter: integer
    maximum number of iterations
    rsize: tuple
    size of the individual pictures

    Return
    -------
    position to move to
    Q-value of the chosen position

    The function takes more inputs and return but the ones mentioned are the relevant ones to functionality.
    """

    state_no = start_location

    current_q, current_action = eps_greedy(q, state_no)

    current_iter = 0

    r_current = 1

    help = 0

    help_stop_iter = constants.iterations*constants.nruns
    interaction_restart_iter = 0

    moving_average_reward = 0 #initialisation

    moving_average_helper = 0  # initialisation


    image_current = start_img

    joint_counter = 10


    while current_iter < max_iter:

        current_iter += 1
        print('now the the new loop started and the current action is: ', current_action)

        no_of_state_visits[state_no] += 1
        image_future = random.randint(0, constants.ntrainimgs - 1)
        while image_future == image_current:
            image_future = random.randint(0, constants.ntrainimgs -1)

        ''' uncomment the next line for a  realistic interaction with speech '''

        # speech_choose_image(current_action)

        display_image(constants.store_grids, 'yellowgrid%s' % current_action, constants.time3)

        '''apply noise to next image, first image noise it applied before start of algorithm'''

        noise_img = sp_noise(constants.get_gameimages, image_future, constants.probabilities[current_action])
        filename = constants.store_gameimages + '%s' % image_future + '.png'
        cv2.imwrite(filename, noise_img)

        time.sleep(constants.time5)


        display_image(constants.store_gameimages, image_future, constants.time2)
        time.sleep(constants.time6)
        result, image = capture_robot_camera_nao(constants.IP, constants.PORT)
        img = Image.fromarray(image)
        img_res = img.crop((constants.left, constants.top, constants.right, constants.bottom))
        filename = constants.store_captured + '%s' % image_future + '.png'
        img_res.save(filename)
        """Closes the window that displays the chosen image in full screen
        """
        done = "pkill eog"
        os.system(done)

        ''' pass current and futute states into the reward function '''


        r_current, energy_current, energy_future = generate_reward(train_images, state_no, current_action, image_current, image_future)
        print('reward: ', r_current, 'current state: ', current_action)
        ''' generate the trust value if help was requested in the previous state '''

        if help == 1:

            joint_counter +=1
            print('When the help is requested the joint counter now is: ', joint_counter)
            helper_reward[0, help_requests] = current_iter
            helper_reward[1, help_requests] = r_current
            print('help reward: ', helper_reward)
            if int(help_requests) >= 3:
                moving_average_helper = (helper_reward[1, help_requests - 2] + helper_reward[1, help_requests - 1] +
                                         helper_reward[1, help_requests]) / 3
                print('moving average helper: ', moving_average_helper)
            if moving_average_helper < constants.mov_avr_tresh and joint_counter > 3 :
                print('####AGENT IS IGNORING PARTNER###', joint_counter)
                # speech_other('Your suggestions are not helpful. I will try by myself!')
                # speech_other2('Ok, try by yourself but maybe I can adapt my strategy and be more helpful if you need help again.')
                interaction_stops += 1
                help_stop_iter = i
                joint_counter = 0

                if int(condition) == 1:
                    condition = 2
                    print('current condition: ', condition)
                elif int(condition) == 2:
                    condition = 1
                    print('current condition: ', condition)

            help_requests += 1

            print('Help requests : ', help_requests)

        help = 0  # reset help value

        ''' add info to saving matrices '''

        total_energy_by_state[state_no] += energy_current
        total_reward += r_current
        total_energy += energy_current
        total_reward_by_state[state_no] += r_current
        cumulative_reward[i] = total_reward
        if i > 3:
            moving_average_reward = ((-cumulative_reward[i-3])+(cumulative_reward[i])) / 3
            print('moving average: ', moving_average_reward)
        cumulative_energy[i] = total_energy

        i += 1
        print('iter:', i)

        ''' update states and image '''

        future_state = current_action
        image_current = image_future

        ''' this codes function behaviour if final state is reached this run '''

        final_states = (0, 10)
        if future_state == final_states[0] or future_state == final_states[1]:

            #print('Final state is reached. It is: ', future_state)

            if future_state ==  final_states[0]:
                r_current = 10
                total_reward += r_current
                total_reward_by_state[future_state] += r_current

                cumulative_reward[i] = total_reward
                td_error = r_current + constants.gamma * q[future_state,future_state] - q[state_no, current_action]
                delta_q = constants.epsilon * (td_error)
                q[state_no, current_action] += delta_q
                print('The current state: ', state_no, 'the chosen action: ', current_action, 'the q value ',  q[state_no, current_action])

                state_no = future_state

                no_of_state_visits[state_no] += 1
            if future_state ==  final_states[1]:
                r_current = -10
                total_reward += r_current
                total_reward_by_state[future_state] += r_current

                cumulative_reward[i] = total_reward
                td_error = r_current + constants.gamma * q[future_state, future_state] - q[state_no, current_action]
                delta_q = constants.epsilon * (td_error)
                q[state_no, current_action] += delta_q
                print('The current state: ', state_no, 'the chosen action: ', current_action, 'the q value ',
                      q[state_no, current_action])

                state_no = future_state

                no_of_state_visits[state_no] += 1


            return future_state, q, total_energy_by_state, no_of_state_visits, total_reward, total_energy, cumulative_reward, cumulative_energy, total_reward_by_state, td_storage, i, help_requests, helper_reward, interaction_stops, interaction_restart, condition
            quit()

        ''' assistant suggesting direction if the reward is negative '''

        if r_current < 0 and i < help_stop_iter:
            help = 1
            #speech_other('I need a tip!')

            if condition == 1: # helpful assistant
                future_action = current_action - 1
                print('Partner suggest move to left: helpful1', current_action, future_action)
                #speech_other2('Move to the left.')
            if condition == 2: # misleading assistant
                future_action = current_action + 1
                print('Partner suggest move to right: unhelpful1', current_action, future_action)
                #speech_other2('Move to the right.')


        elif r_current < 0 and moving_average_reward < constants.mov_avr_tresh and i >= help_stop_iter + 3:
            print('####AGENT IS RECONSIDERING PARTNER###', joint_counter)
            #speech_other('Can I have a tip from your new strategy.')
            interaction_restart += 1
            interaction_restart_iter = i
            joint_counter += 1
            print('The number of interaction restarts is: ', interaction_restart)
            help = 1

            if condition == 1:  # helpful assistant
                future_action = current_action - 1
                #speech_other2('Move to the left.')
                print('partner suggest left: helpful2', current_action, future_action)
            elif condition == 2:  # misleading assistant
                future_action = current_action + 1
                #speech_other2('Move to the right.')
                print('partner suggests right: unhelpful2', current_action, future_action)


            help_stop_iter = constants.iterations * constants.nruns

            ''' or the agent choosing its future action '''

        else:
            future_q, future_action = eps_greedy(q, future_state)
            print('This only prints when agent makes free choice:' , future_action)

        ''' updates other storage matrices '''
        print('after the action selection the future action is: ', future_action)
        td_error = r_current + constants.gamma * q[future_state, future_action] - q[state_no, current_action]
        td_storage[i] = td_error

        delta_q = constants.lr * (td_error)
        q[state_no, current_action] += delta_q

        ''' update states'''

        state_no = future_state

        current_action = future_action
        print('now the future action turned to the current action: ', current_action)




    return future_state, q, total_energy_by_state, no_of_state_visits, total_reward, total_energy, cumulative_reward, cumulative_energy, total_reward_by_state, td_storage, i, help_requests, helper_reward, interaction_stops, interaction_restart, condition


def eps_greedy(q, state):
    """Generates chosen position based on an epsilon greedy policy
        Parameters
    -----------
    q: matrix
    Q-values
    state: integer
    current state
    epsilon: floatE
    chance of choosing random action

    Return
    -------
    action: position to move to
    action_q: Q-value of the chosen position
    """
    value = np.random.uniform(low=0, high=1, size=1)
    state = int(state)

    move_up = int(state + 1)
    move_down = int(state - 1)

    if value >= constants.epsilon:

        action_q = np.max((q[state, move_down], q[state, move_up]))
        if action_q == q[state, move_down]:
            action = move_down
        else:
            action = move_up

        action = np.array(action).flatten()[0]


    else:
        move = np.random.choice([move_up, move_down])
        action_q = q[state, move]
        action = move

        action = np.array(action).flatten()[0]

    return action_q, action

