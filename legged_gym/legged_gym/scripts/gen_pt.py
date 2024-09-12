import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl/rsl_rl/")
os.sys.path.insert(0, rsl_rl_dir)
import torch
import copy
from modules.actor_critic import ActorCritic

# def act_inference(self, observations, observation_history, policy_info={}):
#         """
#         Performs action selection during inference.
#         Args:
#             observations (Tensor): The current observations.
#             observation_history (Tensor): The observation history.
#             privileged_observations (None or Tensor): The privileged observations.
#             policy_info (dict): Dictionary to store policy information.
#         Returns:
#             Tensor: The inferred actions.
#         """
#         latent = self.adaptation_module(observation_history)
#         actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
#         policy_info["latents"] = latent.detach().cpu().numpy()
#         return actions_mean

class PolicyExporter(torch.nn.Module):
    def __init__(self, adaptation_module, actor_body): 
        super().__init__()
        self.adaptation_module = copy.deepcopy(adaptation_module)
        self.actor_body = copy.deepcopy(actor_body)
        self.adaptation_module.cpu()
        self.actor_body.cpu()

    def forward(self, observations_total):
        obs = observations_total[:,:117]
        obs_history = observations_total[:,117:]
        
        latent_e = self.adaptation_module(obs_history)
        actions_mean = self.actor_body(torch.cat((obs, latent_e), dim=-1))
        
        return actions_mean

    def export(self, save_path, model_name):
        # traced_script_module = torch.jit.script(self, observations_total.cpu())
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(save_path + model_name)

# def load(self, path, load_optimizer=True):
#         loaded_dict = torch.load(path, map_location=self.device)
#         self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
#         if load_optimizer:
#             self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
#         self.current_learning_iteration = loaded_dict['iter']
#         return loaded_dict['infos']

if __name__ == '__main__':
    path = 'your log path'
    model_name = 'model_13000.pt'
    loaded_dict = torch.load(path + model_name, map_location='cpu')
    actor_critic = ActorCritic(117, # env.num_obs, 
                               54,  # env.num_previleged_obs, 
                               40*117,  # env.num_obs_history, 
                               12,  
                               init_noise_std = 1.0, # check the correctness with that in config file!!
                               actor_hidden_dims = [512, 256, 128], 
                               critic_hidden_dims = [512, 256, 128],
                               adaptation_hidden_dims = [256, 32],
                               encoder_latent_dims = 18,
                               activation = 'elu').to('cpu')   # env.num_policy_outputs

    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    policyExporter = PolicyExporter(actor_critic.adaptation_module, actor_critic.actor)
    if not os.path.exists(path+'exported/'):
        os.mkdir(path+'exported/')
    policyExporter.export(path+'exported/', model_name)