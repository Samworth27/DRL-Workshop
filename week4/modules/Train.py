import json
import os
from timeit import default_timer as timer
from modules.Helpers import timeString, cleanDict

def runTraining(algorithm, num_epochs, checkpoint_root, last_checkpoint = None, save_interval=None, results_interval=None, eval_interval=None ):

    if last_checkpoint:
        print(f"Loading Checkpoint {last_checkpoint}")
        checkpoint_file = f'{checkpoint_root}/checkpoint_{last_checkpoint:06}/checkpoint-{last_checkpoint}'
        checkpoint = algorithm.restore(checkpoint_file)
    else:
        checkpoint = algorithm.save(checkpoint_root)

    start = timer()

    start_epoch = last_checkpoint if last_checkpoint else 0

    for epoch in range(start_epoch+1, num_epochs+1):

        interim = timer()

        result = algorithm.train()
        current = timer()
        running_time = current - start
        average_time = running_time/(epoch-start_epoch)

        os.system('clear')
        
        min_reward = result["episode_reward_min"]
        max_reward = result["episode_reward_max"]
        mean_reward = result["episode_reward_mean"]
        time_remaining = (num_epochs+1 - epoch)*average_time

        print("------------------------------------------------------------------")
        print(
            f'Running Time: {timeString(running_time)} \naverage completion: {round(average_time,2)}s')
        print(
            f'Estimated Time Remaining: {timeString(time_remaining)}')
        print("------------------------------------------------------------------")
        print(f'finished epoch: {epoch} in {round(current - interim,2)} seconds')
        print(f'Mean Reward: {mean_reward}\n Min: {min_reward} Max: {max_reward}')
        print("------------------------------------------------------------------")

        if save_interval:
            if epoch % save_interval == 0:
                checkpoint = algorithm.save(checkpoint_root)


        if results_interval:
            if epoch % results_interval == 0:
                # checkpoint = ((LAST_CHECKPOINT if LAST_CHECKPOINT else 0) + epoch)
                checkpoint = epoch
                checkpoint_dir = f'./{checkpoint_root}/checkpoint_{checkpoint:06}'
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(f'{checkpoint_dir}/result.json', 'w') as fp:
                    json.dump(cleanDict(result, type(algorithm.config)), fp,  indent=4)
            
            
        if eval_interval:
            if epoch % eval_interval == 0:
                eval = algorithm.evaluate()
                # checkpoint = ((LAST_CHECKPOINT if LAST_CHECKPOINT else 0) + epoch)
                checkpoint = epoch
                checkpoint_dir = f'./{checkpoint_root}/checkpoint_{checkpoint:06}'
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(f'{checkpoint_dir}/eval.json', 'w') as fp:
                    json.dump(cleanDict(result, type(algorithm.config)), fp,  indent=4)