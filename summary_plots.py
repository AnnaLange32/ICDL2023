import numpy as np
import matplotlib.pyplot as plt
import constants

plt.style.use('seaborn-white')

filepath = '/home/anna/nao_trust_2/adaptive_partner/outputs/plot_present/'
conditions = ['adaptive_start_unhelp/'] #['adaptive_start_unhelp/'] random_help_unhelp/
title = ['Trustworthy', 'Non-Trustworthy', 'Random']
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(35,12)) #figsize=(45,12)


cum_rew_arr_0 = np.zeros((constants.repeats, constants.iterations*constants.nruns))

for repeat in np.arange(1, constants.repeats + 1):
    cumulative_reward_0 = np.load(filepath + conditions[0] + '%s_cum_rew.npy' % repeat)

    cumulative_reward_0 = cumulative_reward_0[cumulative_reward_0 != 0]
    cum_rew_arr_0[repeat-1,:len(cumulative_reward_0)] =  cumulative_reward_0

#for repeat in np.arange(1, constants.repeats + 1):
 #   cumulative_reward_1 = np.load(filepath + conditions[1] + '%s_cum_rew.npy' % repeat)

  #  cumulative_reward_1 = cumulative_reward_1[cumulative_reward_1 != 0]
   # cum_rew_arr_1[repeat-1,:len(cumulative_reward_1)] =  cumulative_reward_1

cumulative_average = np.zeros(len(cum_rew_arr_0[0,:]))

#cumulative_average = np.zeros((2, len(cum_rew_arr_1[0,:])))

'''Calculate the average curve, with full run length'''

cumulative_average_v2 = np.zeros(len(cum_rew_arr_0[0,:]))
repeat_holder = np.zeros(constants.repeats)

# Loop through the range of indices (assuming cum_rew_arr_0 is a numpy array)
for i in range(len(cum_rew_arr_0[0, :])):


    # Loop through the range of repeats (assuming constants.repeats is a list)
    for repeat in range(constants.repeats):
        if cum_rew_arr_0[repeat, i] != 0:
            repeat_holder[repeat] = cum_rew_arr_0[repeat, i]

        # Calculate the mean of repeat_holder and assign it to the corresponding index in cumulative_average_v2
    cumulative_average_v2[i] = np.mean(repeat_holder)

cumulative_average[:] = np.mean(cum_rew_arr_0, axis =0)
print(cumulative_average, cumulative_average_v2)
print(cumulative_average)
#cumulative_average[76:] = np.nan

#cumulative_average[92:] = np.nan

for repeat in np.arange(1, constants.repeats + 1):
    cumulative_reward = np.load(filepath + conditions[0] + '%s_cum_rew.npy' % repeat)

    cumulative_reward_plot = cumulative_reward[cumulative_reward != 0]
        #iterations = np.arange(0, len(cumulative_energy_plot), 1)
    iterations = np.arange(0, len(cumulative_reward_plot), 1)


        #plt.plot(cumulative_energy_plot)
    plt.plot(cumulative_reward_plot)

        #plt.title('Cumulative energy')
        #plt.title(title[i], size = 39)
    plt.xlabel('Iteration', fontsize = 50)
    plt.ylabel('Cumulative Reward', fontsize = 50)
    #plt.title(label="Fixed Condition", fontsize = 45)
    plt.tick_params(axis='both', which='major', labelsize=45)
    plt.xlim([0, 160])
    plt.ylim([-90, 250])
    plt.xticks(np.arange(0, 160 , 20.0))
    plt.yticks(np.arange(-50, 250 , 50.0))

        #set(plt.gca(), 'FontSize', 20)
        #plt.savefig('/home/anna/nao_trust_2/multimodal_trust/outputs/output_repeat%s_cum_en.png' % repeat)

#plt.suptitle('Cumulative reward', size = 52, y=0.99)
cumulative_average_plot_v2 = cumulative_average_v2[cumulative_average != 0]
cumulative_average_plot = cumulative_average[cumulative_average != 0]
plt.plot(cumulative_average_plot_v2[:], linewidth = 7)


#plt.plot(cumulative_average_plot[:], linewidth = 5)
plt.savefig('/home/anna/nao_trust_2/adaptive_partner/outputs/plot_present/cum_rew_adapt5.png' % repeat)
plt.show()


#add xlim ylim so no gap to zero

